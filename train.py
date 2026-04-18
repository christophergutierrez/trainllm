import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import _config
cfg = _config.load()

os.environ["HF_HOME"] = str(cfg.hf_home)

import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, standardize_sharegpt
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq

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
    load_in_4bit=True,
    device_map={"": torch.cuda.current_device()},
    attn_implementation="sdpa",
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

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=cfg.training.max_seq_length,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    dataset_kwargs={"skip_prepare_dataset": True},
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

print("Saving final adapter...")
model.save_pretrained(str(OUTPUT_DIR / "final"))
tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))
print("Done.")
