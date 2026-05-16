# DPO Fine-Tuning

Direct Preference Optimization (DPO) is a second training pass that runs on top of an existing SFT adapter. It pushes the model toward chosen completions and away from rejected ones.

## When to use DPO

DPO is the right tool when:
- The model has a strong pretrained prior that SFT alone can't override (e.g., "campaign" always mapping to `/campaigns` instead of `/measurements`).
- The model chains two API calls when a single call suffices — it learned the chaining pattern too aggressively during SFT.
- You can construct clear chosen/rejected pairs where the distinction is unambiguous.

DPO is **not** the right tool when:
- The model simply hasn't seen enough examples of the correct behavior — add more SFT data first.
- The problem is format-level (wrong JSON structure, missing fields) — that's a data quality issue for SFT.
- You'd need hundreds of DPO pairs — at that scale, retraining with better SFT data is usually more effective.

Typical DPO runs are small: 20-100 pairs, 80-100 steps.

## Data format

DPO training data is JSONL with one preference pair per line:

```json
{
  "question": "get measurement 42",
  "chosen": {
    "endpoint": "GET /external/v1/measurements/{id}",
    "params": {"id": 42}
  },
  "rejected": {
    "steps": [
      {"endpoint": "GET /external/v1/measurements", "params": {}},
      {"endpoint": "GET /external/v1/measurements/{id}", "params": {"id": "{{steps.0.id}}"}}
    ]
  }
}
```

- `question`: the user's natural language request.
- `chosen`: the correct API call (single call or chain).
- `rejected`: the wrong behavior the model currently produces.

Both `chosen` and `rejected` are formatted as API call objects (same structure as `api_call` in training data). The script wraps them in fenced JSON code blocks and applies the chat template.

### Constructing good pairs

1. Run eval on the SFT adapter and find the failing cases.
2. For each failure, the model's generated output is the `rejected` completion. The reference answer is the `chosen` completion.
3. Focus on systematic failures (the model makes the same mistake on a class of prompts), not one-off errors.

## Usage

```bash
# Basic run:
DPO_DATA=~/data/dpo.jsonl python train_dpo.py

# With a different config:
TRAINLLM_CONFIG=config.acme.yaml DPO_DATA=~/data/dpo.jsonl python train_dpo.py

# Override hyperparameters:
DPO_BETA=0.2 DPO_MAX_STEPS=80 DPO_LR=3e-5 DPO_DATA=~/data/dpo.jsonl python train_dpo.py

# Explicit SFT adapter path:
SFT_DIR=~/trainLLM/lora/my-adapter/final DPO_DATA=~/data/dpo.jsonl python train_dpo.py
```

## How it works

1. Loads the SFT adapter from `lora/<adapter>/final` (or `SFT_DIR`).
2. Adds a **new LoRA adapter** on top of the SFT adapter. The SFT weights stay frozen — only the DPO LoRA trains.
3. Builds a dataset with three columns per record:
   - `prompt`: system prompt + user question (with `add_generation_prompt=True`)
   - `chosen`: full message sequence through the correct assistant response
   - `rejected`: full message sequence through the wrong assistant response
4. Runs `DPOTrainer` with `ref_model=None`.
5. Saves to `lora/<adapter>/dpo_final/final`.

### The ref_model=None trick

With `ref_model=None` on a PEFT model, DPOTrainer uses the adapter-disabled forward pass as the reference. This means the reference model is the **base model** (Qwen, Llama, etc.), not the SFT adapter.

For targeted corrections this is fine: the DPO signal is clear because the base model's behavior is far from both chosen and rejected, so the preference gradient is strong. The `beta` parameter (KL penalty) prevents the policy from drifting too far from the SFT baseline.

If you need the SFT adapter as the reference (tighter control over how far DPO can deviate from SFT behavior), load a second copy of the SFT adapter and pass it as `ref_model`. This roughly doubles GPU memory usage.

## Hyperparameters

| Parameter | Env var | Default | Notes |
|-----------|---------|---------|-------|
| Beta | `DPO_BETA` | 0.3 | KL penalty. Higher = more conservative (stays closer to reference). Lower = stronger preference signal. |
| Max steps | `DPO_MAX_STEPS` | 100 | DPO datasets are small; 80-100 steps is typical. |
| Learning rate | `DPO_LR` | 5e-5 | Lower than SFT (2e-4) because the correction is targeted. |
| Batch size | `DPO_BATCH` | from config | Inherits SFT config default. |
| Gradient accumulation | `DPO_GRAD_ACCUM` | from config | Inherits SFT config default. |

### Beta tuning

- **0.1**: Aggressive correction. Use when the SFT model's bad behavior is very strong and you want maximum DPO effect. Risk: may overcorrect on adjacent prompts.
- **0.3** (default): Balanced. Good starting point for most corrections.
- **0.5+**: Conservative. Use when you want a gentle nudge without risking regression on other behaviors.

## Output and downstream integration

DPO saves to `lora/<adapter>/dpo_final/final/`, parallel to the SFT adapter at `lora/<adapter>/final/`.

### Merge priority

`merge.py` checks adapters in this order:
1. `lora/<name>/dpo_final/final` (DPO adapter preferred)
2. `lora/<name>/dpo_final`
3. `lora/<name>/final` (SFT fallback)

If a DPO adapter exists, it is used for merging automatically.

### Archive tracking

`archive.py` records `has_dpo: true` in adapter metadata when a `dpo_final` directory exists, so archives reflect whether DPO was applied.

### Evaluating DPO results

After DPO training, serve the adapter and run eval:

```bash
# Start vLLM with the DPO adapter:
vllm serve <base_model> --dtype bfloat16 --enforce-eager --enable-lora \
  --lora-modules my-adapter=~/trainLLM/lora/my-adapter/dpo_final/final \
  --port 8000

# Eval:
MODEL=my-adapter HOLDOUT=~/trainLLM/data/holdout.jsonl python eval.py
```

Or use `cycle.py --skip-train` after manually placing the DPO adapter at the final path.

## Typical workflow

1. Train SFT adapter: `python cycle.py`
2. Review eval report — identify systematic failures.
3. Create DPO pairs targeting those failures (20-100 pairs).
4. Run DPO: `DPO_DATA=~/data/dpo.jsonl python train_dpo.py`
5. Serve and evaluate the DPO adapter.
6. If regressions appear on non-targeted prompts, increase `DPO_BETA`.
