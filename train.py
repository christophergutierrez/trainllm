import argparse
import gc
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import _config

try:
    from transformers import TrainerCallback
    from _callbacks import emit_warning, emit_error  # noqa: E402
except ImportError:
    TrainerCallback = object  # type: ignore[assignment,misc]
    def emit_warning(code, message, detail=None): pass  # type: ignore[misc]
    def emit_error(code, message, detail=None): pass  # type: ignore[misc]


class PlateauDetector(TrainerCallback):
    """Stop early when training loss plateaus; emit convergence stats."""

    def __init__(self, patience_steps: int = 200, min_delta: float = 0.002, min_steps: int = 0):
        self.patience_steps = patience_steps
        self.min_delta = min_delta
        self.min_steps = min_steps
        self.best_loss = float("inf")
        self.best_step = 0
        self.loss_history: list[tuple[int, float, float]] = []  # (step, loss, epoch)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return
        loss = logs["loss"]
        epoch = logs.get("epoch", 0)
        step = state.global_step
        self.loss_history.append((step, loss, epoch))

        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.best_step = step

        stalled_for = step - self.best_step
        if stalled_for >= self.patience_steps and step > args.warmup_steps * 2 and step >= self.min_steps:
            print(f"\n*** EARLY STOP: loss plateaued at {self.best_loss:.4f} "
                  f"(step {self.best_step}), no improvement for {stalled_for} steps ***\n")
            emit_warning("PLATEAU", f"Early stop: loss plateaued at {self.best_loss:.4f} "
                         f"(step {self.best_step}), stalled for {stalled_for} steps",
                         {"best_loss": self.best_loss, "best_step": self.best_step,
                          "stalled_steps": stalled_for})
            control.should_training_stop = True

    def on_train_end(self, args, state, control, **kwargs):
        if not self.loss_history:
            return
        first_step, first_loss, _ = self.loss_history[0]
        last_step, last_loss, last_epoch = self.loss_history[-1]
        total_steps = last_step - first_step
        total_drop = first_loss - last_loss
        rate = total_drop / max(1, total_steps)

        summary = {
            "first_loss": round(first_loss, 5),
            "final_loss": round(last_loss, 5),
            "best_loss": round(self.best_loss, 5),
            "best_step": self.best_step,
            "total_steps": state.global_step,
            "final_epoch": round(last_epoch, 2),
            "convergence_rate": round(rate, 7),
            "stopped_early": state.global_step < args.max_steps,
        }

        # Estimate where loss would hit target thresholds
        if rate > 0:
            for target in [0.03, 0.02, 0.01]:
                if last_loss > target:
                    extra = int((last_loss - target) / rate)
                    summary[f"steps_to_{target}"] = state.global_step + extra

        print(f"\n=== Convergence Summary ===")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        print(f"===========================\n")

        summary["loss_history"] = [[s, l] for s, l, _ in self.loss_history]

        out = Path(args.output_dir) / "convergence.json"
        out.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Convergence stats: {out}")


def _find_latest_checkpoint(output_dir: Path) -> str | None:
    """Find the most recent valid checkpoint for resume."""
    checkpoints = sorted(
        (p for p in output_dir.glob("checkpoint-*") if p.is_dir()),
        key=lambda p: int(p.name.split("-")[1]) if p.name.split("-")[1].isdigit() else 0,
    )
    for ckpt in reversed(checkpoints):
        if (ckpt / "trainer_state.json").exists():
            return str(ckpt)
    return None


def _checkpoint_step(checkpoint: str | None) -> int:
    if checkpoint is None:
        return 0
    path = Path(checkpoint) / "trainer_state.json"
    try:
        state = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return 0
    return int(state.get("global_step") or 0)


def main() -> None:
    # --- Argument parsing (runs before config or model loading) ---
    parser = argparse.ArgumentParser(
        description="Fine-tune an LLM with LoRA using Unsloth."
    )
    parser.add_argument("--base-model", type=str, default=None, metavar="STR",
                        help="Base model name or path (overrides cfg.model)")
    parser.add_argument("--train-data", type=str, default=None, metavar="PATH",
                        help="Training data file (overrides TRAIN_DATA env / cfg.train_data)")
    parser.add_argument("--output-dir", type=str, default=None, metavar="PATH",
                        help="Output directory for LoRA adapter (overrides OUTPUT_DIR env / cfg.lora_dir)")
    parser.add_argument("--max-steps", type=int, default=None, metavar="INT",
                        help="Maximum training steps (overrides MAX_STEPS env / cfg.training.max_steps)")
    parser.add_argument("--adapter-name", type=str, default=None, metavar="STR",
                        help="Adapter name (overrides cfg.adapter_name)")
    parser.add_argument("--config", type=Path, default=None, metavar="PATH",
                        help="Config YAML path (overrides TRAINLLM_CONFIG)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print resolved config and exit without loading any model")
    parser.add_argument("--probe-data", type=Path, default=None, metavar="PATH",
                        help="HumanEval JSONL to probe pass@1 every 250 steps (early quality check)")
    args = parser.parse_args()

    # --- Config loading (after argparse so --help exits cleanly) ---
    cfg = _config.load(args.config)

    # HF_HOME must be set before importing torch/unsloth — they read it at import time.
    os.environ["HF_HOME"] = str(cfg.hf_home)

    # --- Resolve effective settings: CLI flags > env vars > config ---
    MODEL_NAME = args.base_model    if args.base_model   is not None else cfg.model
    DATA_PATH  = Path(args.train_data  if args.train_data  is not None else os.environ.get("TRAIN_DATA",  str(cfg.train_data)))
    OUTPUT_DIR = Path(args.output_dir  if args.output_dir  is not None else os.environ.get("OUTPUT_DIR",  str(cfg.lora_dir)))
    MAX_STEPS  = args.max_steps        if args.max_steps   is not None else int(os.environ.get("MAX_STEPS", str(cfg.training.max_steps)))
    if args.adapter_name is not None:
        cfg.adapter_name = args.adapter_name

    # --- Dry-run: print resolved config and exit without loading any model ---
    if args.dry_run:
        print(f"Model:         {MODEL_NAME}")
        print(f"Adapter name:  {cfg.adapter_name}")
        print(f"Training data: {DATA_PATH}")
        print(f"Output dir:    {OUTPUT_DIR}")
        print(f"Max steps:     {MAX_STEPS}")
        print(f"Optimizer:     {cfg.training.optimizer}")
        sys.exit(0)

    # --- Heavy imports (placed here so HF_HOME is already set) ---
    import torch  # noqa: E402
    from unsloth import FastLanguageModel  # noqa: E402
    from unsloth.chat_templates import get_chat_template, standardize_sharegpt, train_on_responses_only  # noqa: E402
    from datasets import load_dataset  # noqa: E402
    from peft import PeftModel  # noqa: E402
    from trl import SFTTrainer  # noqa: E402
    from transformers import EarlyStoppingCallback, TrainingArguments, DataCollatorForSeq2Seq  # noqa: E402
    from _callbacks import WSDDecayCallback, EventEmitterCallback, CudaCacheFlushCallback, TaskEvalCallback, emit_error, emit_warning  # noqa: E402

    print(f"Model:         {MODEL_NAME}")
    print(f"Adapter name:  {cfg.adapter_name}")
    print(f"Training data: {DATA_PATH}")
    print(f"Output dir:    {OUTPUT_DIR}")
    print(f"Max steps:     {MAX_STEPS}")
    print(f"Optimizer:     {cfg.training.optimizer}")

    # Flush any lingering GPU allocations from previous processes before loading
    # the model. On GB10 unified memory, dead-process allocations can persist.
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    print(f"GPU memory before load: "
          f"{torch.cuda.memory_allocated()/1e9:.2f} GB allocated, "
          f"{torch.cuda.memory_reserved()/1e9:.2f} GB reserved")

    quant_kwargs = {}
    if cfg.training.load_in_fp8:
        quant_kwargs["load_in_4bit"] = False
        quant_kwargs["load_in_fp8"] = True
        print("Quantization:  FP8")
    elif cfg.training.load_in_4bit:
        quant_kwargs["load_in_4bit"] = True
        print("Quantization:  4-bit")
    else:
        quant_kwargs["load_in_4bit"] = False
        print("Quantization:  none (bf16)")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=cfg.training.max_seq_length,
        dtype=None,
        **quant_kwargs,
        device_map={"": torch.cuda.current_device()},
        attn_implementation="sdpa",
        trust_remote_code=True,
    )

    lora_init = cfg.training.lora_init
    if lora_init in ("true", "True"):
        lora_init = True
    elif lora_init in ("false", "False"):
        lora_init = False

    resume_ckpt = _find_latest_checkpoint(OUTPUT_DIR)
    resume_adapter_only = False
    completed_steps = _checkpoint_step(resume_ckpt)

    if resume_ckpt:
        args_path = Path(resume_ckpt) / "training_args.bin"
        try:
            old_args = torch.load(args_path, map_location="cpu", weights_only=False)
            old_optim = getattr(old_args, "optim", None)
            if str(old_optim).split(".")[-1].lower() != cfg.training.optimizer.lower():
                resume_adapter_only = True
                emit_warning(
                    "FRESH_OPTIMIZER",
                    f"Rebuilding optimizer state because checkpoint used {old_optim} "
                    f"and config uses {cfg.training.optimizer}",
                    {"checkpoint": Path(resume_ckpt).name, "checkpoint_optimizer": str(old_optim)},
                )
        except (OSError, RuntimeError, ValueError) as e:
            resume_adapter_only = True
            emit_warning(
                "FRESH_OPTIMIZER",
                f"Could not inspect checkpoint optimizer state; loading adapter weights only: {e}",
                {"checkpoint": Path(resume_ckpt).name},
            )

    if resume_adapter_only and resume_ckpt:
        print(f"Loading adapter weights from: {resume_ckpt}")
        print("Rebuilding optimizer/scheduler state from current config.")
        model = PeftModel.from_pretrained(model, resume_ckpt, is_trainable=True)
    else:
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg.training.lora_rank,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=cfg.training.lora_alpha,
            lora_dropout=cfg.training.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
            use_rslora=cfg.training.use_rslora,
            init_lora_weights=lora_init,
        )

    print("Loading dataset...")
    tokenizer = get_chat_template(tokenizer, chat_template=cfg.chat_template)
    dataset = load_dataset("json", data_files=str(DATA_PATH), split="train")
    dataset = standardize_sharegpt(dataset)

    def format_prompts(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False)
                 for c in convos]
        return {"text": texts}

    dataset = dataset.map(format_prompts, batched=True)
    print(f"Dataset size: {len(dataset)} records")

    if cfg.training.eval_during_training:
        train_dataset = dataset
        if cfg.holdout.exists():
            eval_dataset = load_dataset("json", data_files=str(cfg.holdout), split="train")
            eval_dataset = standardize_sharegpt(eval_dataset)
            eval_dataset = eval_dataset.map(format_prompts, batched=True)
            print(f"Train records: {len(train_dataset)} | Holdout records: {len(eval_dataset)}")
        else:
            split = dataset.train_test_split(test_size=0.05, seed=42)
            train_dataset = split["train"]
            eval_dataset  = split["test"]
            print(f"Train split: {len(train_dataset)} | Eval split: {len(eval_dataset)}")
    else:
        train_dataset = dataset
        eval_dataset  = None

    steps_per_epoch = max(1, len(dataset) // (cfg.training.batch_size * cfg.training.gradient_accumulation_steps))
    plateau = PlateauDetector(patience_steps=200, min_delta=0.002, min_steps=steps_per_epoch)

    callbacks         = [plateau, EventEmitterCallback(), CudaCacheFlushCallback()]
    if cfg.training.eval_during_training:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=4))

    if args.probe_data and args.probe_data.exists():
        import json as _json
        _probe_problems = [_json.loads(l) for l in args.probe_data.read_text().splitlines() if l.strip()][:20]
        print(f"TaskProbe:      {len(_probe_problems)} problems from {args.probe_data}")

        def _probe_fn(probe_model, probe_tokenizer, step):
            from code_verify import clean_completion, verify_humaneval

            active_model = probe_model or model
            active_tokenizer = probe_tokenizer or tokenizer
            active_model.eval()
            passed = 0
            with torch.no_grad():
                for prob in _probe_problems:
                    text = active_tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": "You are an expert Python programmer. Output only code."},
                            {"role": "user", "content": prob["prompt"]},
                        ],
                        tokenize=False, add_generation_prompt=True,
                    )
                    inputs = active_tokenizer(text, return_tensors="pt").to(active_model.device)
                    out = active_model.generate(**inputs, max_new_tokens=256,
                                                do_sample=False,
                                                pad_token_id=active_tokenizer.eos_token_id)
                    completion = active_tokenizer.decode(
                        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                    completion = clean_completion(completion)
                    if verify_humaneval(
                        prob["prompt"], completion, prob["test"],
                        entry_point=prob.get("entry_point", ""),
                    ).passed:
                        passed += 1
            active_model.train()
            return passed / len(_probe_problems)

        callbacks.append(TaskEvalCallback(_probe_fn, probe_every=250, min_step=100))
    actual_lr_sched   = cfg.training.lr_scheduler
    if cfg.training.lr_scheduler == "wsd":
        actual_lr_sched = "constant_with_warmup"
        wsd = WSDDecayCallback(
            stable_ratio=cfg.training.wsd_stable_ratio,
            min_lr_ratio=cfg.training.wsd_min_lr_ratio,
        )
        callbacks.append(wsd)
        print(f"LR schedule:    wsd (stable={cfg.training.wsd_stable_ratio}, "
              f"min_lr={cfg.training.wsd_min_lr_ratio})")

    neftune_alpha = cfg.training.neftune_noise_alpha
    if neftune_alpha and neftune_alpha > 0:
        print(f"NEFTune:        alpha={neftune_alpha}")

    effective_max_steps = MAX_STEPS
    if resume_adapter_only and completed_steps:
        effective_max_steps = max(1, MAX_STEPS - completed_steps)
        print(f"Remaining train steps: {effective_max_steps} ({completed_steps}/{MAX_STEPS} already completed)")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=cfg.training.max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        dataset_kwargs={"skip_prepare_dataset": True},
        callbacks=callbacks,
        neftune_noise_alpha=neftune_alpha if neftune_alpha and neftune_alpha > 0 else None,
        args=TrainingArguments(
            per_device_train_batch_size=cfg.training.batch_size,
            gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
            warmup_steps=cfg.training.warmup_steps,
            max_steps=effective_max_steps,
            learning_rate=cfg.training.learning_rate,
            bf16=True,
            logging_steps=10,
            optim=cfg.training.optimizer,
            weight_decay=cfg.training.weight_decay,
            lr_scheduler_type=actual_lr_sched,
            seed=42,
            output_dir=str(OUTPUT_DIR),
            save_steps=cfg.training.save_steps,
            save_total_limit=cfg.training.save_total_limit,
            eval_strategy="steps" if eval_dataset is not None else "no",
            eval_steps=cfg.training.save_steps if eval_dataset is not None else None,
            load_best_model_at_end=eval_dataset is not None,
            metric_for_best_model="eval_loss" if eval_dataset is not None else None,
            greater_is_better=False if eval_dataset is not None else None,
        ),
    )

    if cfg.training.train_on_responses_only:
        print("Loss masking:   assistant tokens only")
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )

    if resume_ckpt:
        if resume_adapter_only:
            emit_warning("RESUMING", f"Loaded adapter weights from {Path(resume_ckpt).name}")
        else:
            emit_warning("RESUMING", f"Resuming from {Path(resume_ckpt).name}")
            print(f"Resuming from: {resume_ckpt}")

    print("Starting training...")
    try:
        trainer.train(resume_from_checkpoint=None if resume_adapter_only else resume_ckpt)
    except torch.cuda.OutOfMemoryError as e:
        step = plateau.loss_history[-1][0] if plateau.loss_history else 0
        emit_error("OOM", f"CUDA out of memory at step {step}", {"step": step, "error": str(e)})
        print(f"FATAL: OOM at step {step}")
        gc.collect()
        torch.cuda.empty_cache()
        sys.exit(137)
    except Exception as e:
        step = plateau.loss_history[-1][0] if plateau.loss_history else 0
        emit_error("TRAIN_CRASH", str(e), {"step": step, "type": type(e).__name__})
        raise

    if plateau.loss_history:
        actual_steps = plateau.loss_history[-1][0]
        target_steps = effective_max_steps
        if actual_steps < target_steps:
            print(f"Training stopped early at step {actual_steps}/{target_steps}")

    print("Saving final adapter...")
    model.save_pretrained(str(OUTPUT_DIR / "final"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))

    print("Releasing GPU memory...")
    del trainer
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    print("Done.")


if __name__ == "__main__":
    main()
