#!/usr/bin/env python3
"""
SimPO fine-tuning pass on top of a trained SFT adapter.

Loads the SFT adapter from lora_dir/final, runs CPOTrainer (SimPO mode) to push
the model toward chosen completions and away from rejected ones, and saves the
result to lora_dir/dpo_final.

SimPO (Simple Preference Optimization) uses a length-normalized reward and a
margin term (simpo_gamma) instead of a reference model, which makes it more
memory-efficient than DPO and better-suited to small LoRA adapters.

Typical use: kill a strong chaining prior that SFT alone can't override.

Usage:
    # Train on a single endpoint's DPO pairs
    DPO_DATA=~/tmp/acme/measurements/dpo.jsonl python train_dpo.py

    # With a custom config
    TRAINLLM_CONFIG=config.acme.yaml DPO_DATA=.../dpo.jsonl python train_dpo.py

    # Override hyperparams
    SIMPO_BETA=2.0 SIMPO_GAMMA=1.0 DPO_MAX_STEPS=80 DPO_LR=3e-5 DPO_DATA=.../dpo.jsonl python train_dpo.py
"""

import gc
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import _config


def main() -> None:
    cfg = _config.load()

    os.environ["HF_HOME"] = str(cfg.hf_home)

    import torch
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    from datasets import Dataset
    from trl import CPOTrainer, CPOConfig

    # ── Hyperparams (env overrides for one-offs) ──────────────────────────────

    SFT_DIR   = Path(os.environ.get("SFT_DIR",      str(cfg.final_dir)))
    DPO_DATA  = Path(os.environ.get("DPO_DATA",     str(cfg.data_dir / "dpo.jsonl")))
    OUTPUT_DIR = cfg.lora_dir / "dpo_final"

    SIMPO_GAMMA = float(os.environ.get("SIMPO_GAMMA", "1.0"))
    SIMPO_BETA  = float(os.environ.get("SIMPO_BETA",  "2.0"))
    MAX_STEPS   = int(os.environ.get("DPO_MAX_STEPS",   "100"))
    LR          = float(os.environ.get("DPO_LR",        "5e-5"))
    BATCH       = int(os.environ.get("DPO_BATCH",       str(cfg.training.batch_size)))
    GRAD_ACCUM  = int(os.environ.get("DPO_GRAD_ACCUM", str(cfg.training.gradient_accumulation_steps)))

    print(f"SFT adapter:   {SFT_DIR}")
    print(f"SimPO data:    {DPO_DATA}")
    print(f"Output dir:    {OUTPUT_DIR}")
    print(f"Beta:          {SIMPO_BETA}  |  Gamma: {SIMPO_GAMMA}  |  Max steps: {MAX_STEPS}  |  LR: {LR}")

    if not DPO_DATA.exists():
        sys.exit(f"DPO data not found: {DPO_DATA}\nSet DPO_DATA env var to the correct path.")

    if not SFT_DIR.exists():
        sys.exit(f"SFT adapter not found: {SFT_DIR}\nRun train.py first, or set SFT_DIR.")

    # ── Load SFT adapter ──────────────────────────────────────────────────────

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(SFT_DIR),
        max_seq_length=cfg.training.max_seq_length,
        dtype=None,
        load_in_4bit=cfg.training.load_in_4bit,
        device_map={"": torch.cuda.current_device()},
    )

    tokenizer = get_chat_template(tokenizer, chat_template=cfg.chat_template)

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
    )

    # ── Data preparation ──────────────────────────────────────────────────────

    from prepare_data import DEFAULT_ORG, build_system_prompt  # noqa: E402

    ORG_NAME = os.environ.get("TRAINLLM_ORG", DEFAULT_ORG)
    SYSTEM_PROMPT = build_system_prompt(ORG_NAME)

    def format_response(api_call: dict) -> str:
        return "```json\n" + json.dumps(api_call, indent=2) + "\n```"

    def make_messages(question: str, response: dict) -> list[dict]:
        return [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": question},
            {"role": "assistant", "content": format_response(response)},
        ]

    def apply_template(messages: list[dict]) -> str:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    def prompt_only(question: str) -> str:
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": question},
        ]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    print(f"\nLoading DPO pairs from {DPO_DATA}...")
    raw = []
    with open(DPO_DATA) as f:
        for line in f:
            line = line.strip()
            if line:
                raw.append(json.loads(line))

    print(f"  {len(raw)} pairs")

    dataset = Dataset.from_dict({
        "prompt":   [prompt_only(r["question"])                         for r in raw],
        "chosen":   [apply_template(make_messages(r["question"], r["chosen"]))   for r in raw],
        "rejected": [apply_template(make_messages(r["question"], r["rejected"])) for r in raw],
    })

    # ── Train ─────────────────────────────────────────────────────────────────

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    trainer = CPOTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=CPOConfig(
            loss_type="simpo",
            cpo_alpha=0.0,
            simpo_gamma=SIMPO_GAMMA,
            beta=SIMPO_BETA,
            max_steps=MAX_STEPS,
            per_device_train_batch_size=BATCH,
            gradient_accumulation_steps=GRAD_ACCUM,
            warmup_steps=min(10, MAX_STEPS // 10),
            learning_rate=LR,
            bf16=True,
            logging_steps=10,
            optim="adamw_torch",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            output_dir=str(OUTPUT_DIR),
            max_length=cfg.training.max_seq_length,
            max_prompt_length=cfg.training.max_seq_length // 2,
        ),
    )

    print(f"\nStarting SimPO training ({len(raw)} pairs, {MAX_STEPS} steps, beta={SIMPO_BETA}, gamma={SIMPO_GAMMA})...")
    trainer.train()

    print("\nSaving DPO adapter...")
    model.save_pretrained(str(OUTPUT_DIR / "final"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))

    print("Releasing GPU memory...")
    del trainer
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    print(f"Done → {OUTPUT_DIR / 'final'}")


if __name__ == "__main__":
    main()
