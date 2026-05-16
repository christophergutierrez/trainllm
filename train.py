import gc
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import _config
cfg = _config.load()

# HF_HOME must be set before importing torch/unsloth — they read it at import time.
os.environ["HF_HOME"] = str(cfg.hf_home)

import torch  # noqa: E402
from unsloth import FastLanguageModel  # noqa: E402
from unsloth.chat_templates import get_chat_template, standardize_sharegpt  # noqa: E402
from datasets import load_dataset  # noqa: E402
from trl import SFTTrainer  # noqa: E402
from transformers import TrainingArguments, TrainerCallback, DataCollatorForSeq2Seq  # noqa: E402


class PlateauDetector(TrainerCallback):
    """Stop early when training loss plateaus; emit convergence stats."""

    def __init__(self, patience_steps: int = 200, min_delta: float = 0.002):
        self.patience_steps = patience_steps
        self.min_delta = min_delta
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
        if stalled_for >= self.patience_steps and step > args.warmup_steps * 2:
            print(f"\n*** EARLY STOP: loss plateaued at {self.best_loss:.4f} "
                  f"(step {self.best_step}), no improvement for {stalled_for} steps ***\n")
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

        out = Path(args.output_dir) / "convergence.json"
        out.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Convergence stats: {out}")

def main() -> None:
    MODEL_NAME = cfg.model
    DATA_PATH  = Path(os.environ.get("TRAIN_DATA",  str(cfg.train_data)))
    OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR",  str(cfg.lora_dir)))
    MAX_STEPS  = int(os.environ.get("MAX_STEPS",    str(cfg.training.max_steps)))

    print(f"Model:         {MODEL_NAME}")
    print(f"Adapter name:  {cfg.adapter_name}")
    print(f"Training data: {DATA_PATH}")
    print(f"Output dir:    {OUTPUT_DIR}")
    print(f"Max steps:     {MAX_STEPS}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=cfg.training.max_seq_length,
        dtype=None,
        load_in_4bit=cfg.training.load_in_4bit,
        device_map={"": torch.cuda.current_device()},
        attn_implementation="sdpa",
        trust_remote_code=True,
    )

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

    plateau = PlateauDetector(patience_steps=200, min_delta=0.002)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=cfg.training.max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        dataset_kwargs={"skip_prepare_dataset": True},
        callbacks=[plateau],
        args=TrainingArguments(
            per_device_train_batch_size=cfg.training.batch_size,
            gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
            warmup_steps=cfg.training.warmup_steps,
            max_steps=MAX_STEPS,
            learning_rate=cfg.training.learning_rate,
            bf16=True,
            logging_steps=10,
            optim="adamw_torch",        # adamw_8bit broken on CUDA 13
            weight_decay=cfg.training.weight_decay,
            lr_scheduler_type=cfg.training.lr_scheduler,
            seed=42,
            output_dir=str(OUTPUT_DIR),
            save_steps=cfg.training.save_steps,
            save_total_limit=cfg.training.save_total_limit,
        ),
    )

    print("Starting training...")
    trainer.train()

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
