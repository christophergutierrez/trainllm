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
except ImportError:
    TrainerCallback = object  # type: ignore[assignment,misc]


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
    parser.add_argument("--dry-run", action="store_true",
                        help="Print resolved config and exit without loading any model")
    args = parser.parse_args()

    # --- Config loading (after argparse so --help exits cleanly) ---
    cfg = _config.load()

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
    from trl import SFTTrainer  # noqa: E402
    from transformers import TrainingArguments, DataCollatorForSeq2Seq  # noqa: E402
    from _callbacks import WSDDecayCallback, EventEmitterCallback, emit_error, emit_warning  # noqa: E402

    print(f"Model:         {MODEL_NAME}")
    print(f"Adapter name:  {cfg.adapter_name}")
    print(f"Training data: {DATA_PATH}")
    print(f"Output dir:    {OUTPUT_DIR}")
    print(f"Max steps:     {MAX_STEPS}")
    print(f"Optimizer:     {cfg.training.optimizer}")

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
        split = dataset.train_test_split(test_size=0.05, seed=42)
        train_dataset = split["train"]
        eval_dataset  = split["test"]
        print(f"Train split: {len(train_dataset)} | Eval split: {len(eval_dataset)}")
    else:
        train_dataset = dataset
        eval_dataset  = None

    steps_per_epoch = max(1, len(dataset) // (cfg.training.batch_size * cfg.training.gradient_accumulation_steps))
    plateau = PlateauDetector(patience_steps=200, min_delta=0.002, min_steps=steps_per_epoch)

    callbacks         = [plateau, EventEmitterCallback()]
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
            max_steps=MAX_STEPS,
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
        ),
    )

    if cfg.training.train_on_responses_only:
        print("Loss masking:   assistant tokens only")
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )

    # Check for a valid checkpoint to resume from
    resume_ckpt = _find_latest_checkpoint(OUTPUT_DIR)
    if resume_ckpt:
        emit_warning("RESUMING", f"Resuming from {Path(resume_ckpt).name}")
        print(f"Resuming from: {resume_ckpt}")

    print("Starting training...")
    try:
        trainer.train(resume_from_checkpoint=resume_ckpt)
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
        if actual_steps < MAX_STEPS:
            print(f"Training stopped early at step {actual_steps}/{MAX_STEPS}")

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
