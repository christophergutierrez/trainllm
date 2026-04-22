# trainLLM

A general-purpose QLoRA fine-tuning pipeline for large language models. Trains a LoRA adapter on top of any Hugging Face base model, serves it via vLLM, and evaluates it against a holdout set — all from a single config file.

## Quick start

1. **Configure** — edit `config.yaml` with your model, adapter name, and data paths.
2. **Prepare data** — place a ShareGPT-format JSONL at `data.train` and a holdout JSONL at `data.holdout` (see [Data formats](#data-formats)).
3. **Run** — execute the full cycle:

```bash
python ~/trainLLM/cycle.py
```

This runs: backup → train → serve (vLLM) → eval (fine-tuned + base) → report.

## Requirements

- Python 3.11+
- [Unsloth Studio](https://github.com/unslothai/unsloth) installed at `paths.unsloth_python`
- [vLLM](https://github.com/vllm-project/vllm) available on `PATH`
- PyYAML: `pip install pyyaml`
- `openai` Python package (used to talk to the vLLM OpenAI-compatible API): `pip install openai`
- NVIDIA GPU (pipeline is tuned for Blackwell/GB10; see [Hardware notes](#hardware-notes))

## Configuration

All settings live in `config.yaml`. Scripts read it at startup; env vars override specific fields for one-offs.

```yaml
model: Qwen/Qwen2.5-Coder-14B-Instruct   # any HuggingFace model ID
adapter_name: my-adapter                  # LoRA module name in vLLM; drives lora/ path
chat_template: qwen-2.5                   # Unsloth template name (see below)

paths:
  base_dir: ~/trainLLM
  hf_home: ~/trainLLM/models/hf
  unsloth_python: ~/.unsloth/studio/unsloth_studio/bin/python3

data:
  train: ~/trainLLM/data/training.jsonl
  holdout: ~/trainLLM/data/holdout.jsonl

training:
  max_seq_length: 2048
  lora_rank: 16
  lora_alpha: 32
  lora_dropout: 0
  batch_size: 2
  gradient_accumulation_steps: 4
  warmup_steps: 50
  max_steps: 2000
  learning_rate: 2.0e-4
  weight_decay: 0.01
  lr_scheduler: cosine
  save_steps: 500
  save_total_limit: null   # null = keep all checkpoints, required for best-checkpoint selection

vllm:
  port: 8000
  gpu_memory_utilization: 0.85

timeouts:
  train_silence: 1800    # seconds; kills training if no output; 1800 accommodates first-time model download
  vllm_startup: 900      # seconds to wait for /v1/models to become ready
  vllm_poll_interval: 30
  eval_timeout: 3600
```

**Supported `chat_template` values** (Unsloth names): `qwen-2.5`, `llama-3.1`, `llama-3.2`, `gemma-it`, `chatml`, `mistral`, `phi-3`. Match this to your base model family.

**Env var overrides** (for one-offs without editing config):

| Variable | Overrides |
|----------|-----------|
| `TRAIN_DATA` | `data.train` |
| `OUTPUT_DIR` | LoRA output directory |
| `MAX_STEPS` | `training.max_steps` |
| `MODEL` | adapter/model name for eval |
| `HOLDOUT` | `data.holdout` |
| `VLLM_URL` | vLLM server URL |

## Data formats

### Training data (ShareGPT JSONL)

Unsloth requires ShareGPT format. Each line is a JSON object:

```json
{"conversations": [
  {"from": "human", "value": "Write a function that ..."},
  {"from": "gpt",   "value": "def my_function(): ..."}
]}
```

Additional fields are ignored. `cycle.py` validates the format before training starts.

### Holdout data (eval JSONL)

Each line is a JSON object with an OpenAI-style `messages` array:

```json
{
  "id": "example-001",
  "label": "descriptive name shown in reports",
  "messages": [
    {"role": "user",      "content": "Write a function that ..."},
    {"role": "assistant", "content": "def my_function(): ..."}
  ],
  "conventions_tested": ["optional", "tags"],
  "source_file": "optional/path/for/reference"
}
```

Only `messages` is required. `id` defaults to the record index if omitted. `conventions_tested` enables per-convention score breakdowns in the report.

## Usage

### Full cycle

```bash
python ~/trainLLM/cycle.py
```

### Skip steps

```bash
# Serve and eval only (no training):
python ~/trainLLM/cycle.py --skip-train

# Eval only (server already running):
python ~/trainLLM/cycle.py --skip-train --skip-serve

# Skip base model eval (faster — fine-tuned only):
python ~/trainLLM/cycle.py --skip-base-eval
```

### Versioned data files

```bash
# Uses data/training_20260416.jsonl + data/holdout_20260416.jsonl:
python ~/trainLLM/cycle.py --version 20260416
```

### Step overrides

```bash
# Quick 300-step canary run to validate a data change before committing to a full cycle:
python ~/trainLLM/cycle.py --canary

# Explicit step count:
python ~/trainLLM/cycle.py --steps 500

# Override holdout file:
python ~/trainLLM/cycle.py --holdout ~/trainLLM/data/holdout_sentinel.jsonl
```

### Sizing a run (canary → full)

Use the canary to find the right `max_steps` before committing GPU hours to a full run:

```bash
# 1. Run a cheap 300-step canary (~35 min):
python ~/trainLLM/cycle.py --canary
```

At the end of the training step, the log prints a step estimate:

```
[INFO] Step estimate: 300 steps → loss 1.821→0.934 (0.00295/step).
       To reach 0.6: ~414 more steps. Suggested: --steps 800
```

```bash
# 2. Run the full cycle with the suggested step count:
python ~/trainLLM/cycle.py --steps 800
```

The estimate uses linear extrapolation from the canary loss curve, so treat it as a starting point rather than a guarantee — loss curves are not truly linear, and data diversity matters as much as step count. If the model is already below 0.6 loss at 300 steps, the canary output says so and the full run may not need many more steps.

### Run scripts directly

```bash
# Train only:
TRAIN_DATA=~/trainLLM/data/training.jsonl python ~/trainLLM/train.py

# Eval only (vLLM must be running):
MODEL=my-adapter HOLDOUT=~/trainLLM/data/holdout.jsonl python ~/trainLLM/eval.py

# Base model eval:
MODEL=Qwen/Qwen2.5-Coder-14B-Instruct python ~/trainLLM/eval.py
```

## Output

| Path | Contents |
|------|----------|
| `lora/<adapter_name>/final/` | Trained LoRA adapter (safetensors + tokenizer) |
| `lora/<adapter_name>/final-v<date>/` | Backup of previous adapter before each run |
| `evals/<timestamp>_<model>.md` | Human-readable eval report |
| `evals/<timestamp>_<model>.json` | Raw eval data for programmatic use |
| `evals/<timestamp>_<model>_synth_status.yaml` | Handoff status for reposynth (emitted by step 5b) |
| `logs/cycle_<timestamp>.log` | Full cycle log |

## Interpreting results

- **Fine-tuned avg > base avg by ≥0.05** — training is helping.
- **Fine-tuned avg < base avg** — something is wrong: check data format, lower LR, fewer steps.
- **Delta < 0.05** — fine-tuning has no effect: check that vLLM is serving the adapter, not just the base model.
- **avg < 0.35 with positive delta** — training is helping but undertrained: increase `max_steps` or add data.
- **Final loss > 1.0** — undertrained; increase `max_steps`.
- **Final loss < 0.15** — likely overfit; check holdout scores.

Similarity scores use token-level sequence matching (0.0–1.0). Bands: Excellent ≥0.8, Good ≥0.6, Partial ≥0.4, Poor <0.4.

## Hardware notes

The pipeline is configured for NVIDIA Blackwell (GB10) with 128 GB LPDDR5X:

- `--enforce-eager` is set on the vLLM command to prevent a `torch.compile` hang on Blackwell.
- `gpu_memory_utilization: 0.85` leaves headroom for LoRA adapter pages.
- `optim: adamw_torch` — `adamw_8bit` is broken on CUDA 13.
- 14B model at 4-bit fits comfortably; larger models may require reducing batch size or sequence length.

## Directory structure

```
trainLLM/
├── config.yaml          # all configuration
├── _config.py           # shared config loader (imported by all scripts)
├── train.py             # QLoRA training via Unsloth
├── eval.py              # holdout evaluation via vLLM
├── cycle.py             # orchestrator: backup → train → serve → eval → report
├── data/                # training and holdout JSONL files
├── models/hf/           # HuggingFace model cache (HF_HOME)
├── lora/
│   └── <adapter_name>/
│       ├── final/       # current trained adapter
│       └── final-v*/    # timestamped backups
├── evals/               # eval reports (.md + .json)
└── logs/                # cycle logs and PID files
```

## See also

- [Architecture](docs/architecture.md) — detailed design notes on the pipeline, training approach, and evaluation.
