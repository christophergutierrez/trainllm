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
- Unsloth Studio installed at `paths.unsloth_python`
- vLLM available on `PATH`
- PyYAML: `pip install pyyaml`
- `openai` Python package (used to talk to the vLLM OpenAI-compatible API): `pip install openai`
- NVIDIA GPU (pipeline is tuned for Blackwell/GB10; see [Hardware notes](#hardware-notes))

## Configuration

All settings live in `config.yaml`. Scripts read it at startup; env vars override specific fields for one-offs.

```yaml
model: Qwen/Qwen2.5-Coder-14B-Instruct   # any HuggingFace model ID
adapter_name: my-adapter                  # LoRA module name in vLLM; drives lora/ path
chat_template: qwen-2.5                   # Unsloth template name (see below)
runtime: vllm                             # "vllm" (default) or "external" (user-managed server)

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

**`cycle.py` / `train.py` / `eval.py`**

| Variable | Overrides |
|----------|-----------|
| `TRAINLLM_CONFIG` | Path to config YAML (default: `config.yaml` in repo root) |
| `TRAIN_DATA` | `data.train` |
| `OUTPUT_DIR` | LoRA output directory |
| `MAX_STEPS` | `training.max_steps` |
| `MODEL` | adapter/model name for eval |
| `HOLDOUT` | `data.holdout` |
| `VLLM_URL` | vLLM server URL |

**`prepare_data.py`**

| Variable | Effect |
|----------|--------|
| `TRAINLLM_ORG` | Organization label for generated system prompts (default: `acme`) |
| `TRAINLLM_PROMPT_STYLE` | System prompt style: `conversational` (default) or `structural` |

**`train_dpo.py`**

| Variable | Default | Effect |
|----------|---------|--------|
| `DPO_DATA` | `data/dpo.jsonl` | Path to DPO pairs JSONL |
| `SFT_DIR` | `lora/<adapter>/final` | Path to trained SFT adapter |
| `DPO_BETA` | `0.3` | KL penalty coefficient |
| `DPO_MAX_STEPS` | `100` | Training steps |
| `DPO_LR` | `5e-5` | Learning rate |
| `DPO_BATCH` | from config | Per-device batch size |
| `DPO_GRAD_ACCUM` | from config | Gradient accumulation steps |

**`llm_judge.py`**

| Variable | Default | Effect |
|----------|---------|--------|
| `ANTHROPIC_API_KEY` | — | If set, uses the Anthropic SDK; otherwise shells out to `claude -p` |
| `JUDGE_MODEL` | `claude-haiku-4-5-20251001` | Claude model for judging |
| `JUDGE_CODEBASE` | `acme` | Loads rubric from `rubrics/<name>.txt` if it exists; falls back to a generic rubric |
| `JUDGE_RUBRIC` | — | Path to a custom rubric file (overrides `JUDGE_CODEBASE` lookup) |

## Data formats

### Training data (ShareGPT JSONL)

Unsloth requires ShareGPT format. Each line is a JSON object:

```json
{"conversations": [
  {"from": "system", "value": "You are an API assistant ..."},
  {"from": "human",  "value": "Write a function that ..."},
  {"from": "gpt",    "value": "def my_function(): ..."}
]}
```

The system turn is optional but `prepare_data.py` emits one by default. Additional fields are ignored. `cycle.py` validates the format before training starts.

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
| `lora/<adapter_name>/final/result.json` | Artifact contract for downstream consumers (see below) |
| `lora/<adapter_name>/final-v<date>/` | Backup of previous adapter before each run |
| `merged/<name>/result.json` | Artifact contract for merged models |
| `evals/<timestamp>_<model>.md` | Human-readable eval report |
| `evals/<timestamp>_<model>.json` | Raw eval data for programmatic use |
| `evals/<timestamp>_<model>_synth_status.yaml` | Handoff status for reposynth (emitted by step 5b) |
| `logs/cycle_<timestamp>.log` | Full cycle log |

### `result.json` (artifact contract)

`cycle.py` emits a `result.json` alongside the trained adapter. This is the interface file that downstream systems (e.g., model-control-plane's `import_adapter.py`) read to import an adapter into a serving registry.

```json
{
  "adapter_name": "my-adapter",
  "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
  "adapter_path": "/home/user/trainLLM/lora/my-adapter/final",
  "format": "peft-lora",
  "lora_rank": 16,
  "lora_alpha": 32,
  "max_seq_length": 2048,
  "eval_avg_score": 0.72,
  "training_steps": 800,
  "final_loss": 0.45,
  "timestamp": "2026-05-07T10:30:00Z"
}
```

For merged models, `format` is `"merged-full"` and includes a `merge_config` block. trainLLM is not aware of the registry or serving infrastructure — it only emits this file.

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
├── config.yaml                # all configuration
├── config.example.yaml        # annotated config template
├── config.acme.yaml           # Acme API fine-tuning config
├── _config.py                 # shared config loader (imported by all scripts)
├── cycle.py                   # orchestrator: backup → train → serve → eval → report
├── train.py                   # QLoRA SFT training via Unsloth
├── train_dpo.py               # DPO fine-tuning on top of an SFT adapter
├── eval.py                    # holdout evaluation via vLLM
├── eval_prompt_baseline.py    # one-off: base model + system prompt eval
├── merge.py                   # DARE-TIES model merging via mergekit
├── archive.py                 # archive adapter weights as versioned tarballs
├── clean_weights.py           # remove weights after archiving
├── llm_judge.py               # LLM-as-judge rescoring (Claude Haiku)
├── prepare_data.py            # convert endpoint data → ShareGPT + holdout splits
├── emit_synth_status.py       # emit synth_status.yaml for reposynth handoff
├── endpoint_runner.py         # per-endpoint adapter automation
├── rubrics/                   # LLM judge rubric files (per codebase)
│   └── acme.txt               # Domain-specific rubric for the acme API codebase
├── docs/
│   └── architecture.md        # detailed design notes
├── data/                      # training and holdout JSONL files
├── models/hf/                 # HuggingFace model cache (HF_HOME)
├── lora/
│   └── <adapter_name>/
│       ├── final/             # current trained adapter
│       └── final-v*/          # timestamped backups
├── merged/                    # DARE-TIES merged full models
├── archive/                   # versioned tarball backups (.tar.gz)
├── evals/                     # eval reports (.md + .json)
└── logs/                      # cycle logs and PID files
```

## Model merging (DARE-TIES)

Adapters can be merged into a single full model via DARE-TIES, eliminating LoRA adapter swaps at inference time. This is useful when multi-step API calls chain through multiple endpoints — a merged model shares its KV cache across steps, making chained calls sublinear.

### Config

Add an optional `merge:` block to `config.yaml`:

```yaml
merge:
  method: dare_ties
  density: 0.9          # fraction of task vector parameters retained (0.9 = keep 90%)
  weight: 1.0           # default per-model weight
  normalize: true
  output_dir: ~/git_home/trainLLM/merged/my-merge
  adapters:             # which adapters to merge (or "all")
    - name: my-api-audiences
      weight: 1.0
    - name: my-api-audience
      weight: 1.0
```

### Standalone merge

```bash
# Merge specific adapters (no config change needed):
python3 merge.py --adapters my-api-measurements,my-api-measurement --density 0.9

# Dry run — print mergekit config without running:
python3 merge.py --adapters my-api-measurements,my-api-measurement --dry-run

# Merge and clean up intermediate unloaded models:
python3 merge.py --adapters my-api-audiences,my-api-audience --density 0.9 --clean
```

### Merge via cycle.py

```bash
# Train, then merge, then eval the merged model:
python3 cycle.py --merge --merge-adapters my-api-audiences,my-api-audience

# Merge and eval only (skip training):
python3 cycle.py --merge-only --merge-adapters my-api-audiences,my-api-audience

# Override density:
python3 cycle.py --merge-only --merge-density 0.95
```

### Serving a merged model

A merged model is a full HuggingFace model (not a LoRA adapter). Serve it directly:

```bash
vllm serve ~/git_home/trainLLM/merged/default --dtype bfloat16 --enforce-eager --port 8000 \
  --served-model-name my-merged
```

### Density tuning

`density` controls how aggressively DARE prunes each adapter's task vector:

| Density | Effect |
|---------|--------|
| 0.95 | Conservative — keeps almost everything, minimal quality loss |
| 0.9 | Recommended default for LoRA-based merges |
| 0.5 | Aggressive — works for models with diverse capabilities, too destructive for similar LoRA adapters |

LoRA adapters with low rank (e.g., 16) have sparse task vectors — aggressive pruning destroys signal. Start at 0.9 and tune down only if the merge produces a model that's too large or if the adapters are highly diverse.

### When to merge vs. combined training

DARE-TIES is most useful when merging models with **different capabilities** (e.g., code generation + math reasoning). For multiple adapters doing the **same task** on different data (like your API endpoints), training a single adapter on combined data often produces better results with no quality tradeoff.

## Archiving and cleanup

Before major changes (swapping the base model, pruning old experiments), archive adapter weights as versioned tarballs.

### Archive weights

```bash
# Archive all adapters with a descriptive tag:
python3 archive.py --tag pre-nemotron --include-merged

# Archive specific adapters only:
python3 archive.py --adapters my-adapter,my-api-audiences

# List existing archives:
python3 archive.py --list
```

Archives are saved to `archive/` (gitignored) as compressed tarballs named `<base_model>_<tag>_<timestamp>.tar.gz`. Each tarball includes:
- Adapter weights (final/ and versioned backups)
- Training configs that produced each adapter
- Most recent eval scores and reports
- A `manifest.json` with base model, git commit, data stats, and per-adapter metadata

Inspect an archive without extracting:

```bash
tar xzf archive/<name>.tar.gz manifest.json -O | python3 -m json.tool
```

### Clean up after archiving

```bash
# Dry run — show what would be removed:
python3 clean_weights.py

# Actually remove (only works if an archive exists):
python3 clean_weights.py --confirm
```

`clean_weights.py` verifies every adapter is present in the latest archive before deleting anything. It also cleans up loose directories in `archive/` and intermediate merged model artifacts.

## Additional scripts

### `train_dpo.py` — DPO fine-tuning

Runs a Direct Preference Optimization pass on top of an existing SFT adapter. Useful for correcting strong priors that SFT alone can't override (e.g., chained multi-step responses when a single call suffices). See the script's docstring for usage and hyperparameter overrides.

### `llm_judge.py` — LLM-as-judge rescoring

Rescores an existing eval JSON using Claude Haiku as a semantic judge. The similarity metric (`difflib.SequenceMatcher`) undercounts correct-but-differently-worded code; the judge metric measures correctness and convention adherence instead.

The rubric is loaded from `rubrics/<JUDGE_CODEBASE>.txt` if the file exists. If not, a generic format-agnostic rubric is used. Set `JUDGE_RUBRIC=/path/to/rubric.txt` to use a fully custom rubric. The shipped `rubrics/acme.txt` contains Go-specific scoring criteria; for non-Go output (e.g., JSON API calls), either omit the rubric file or create a domain-specific one.

### `prepare_data.py` — API training data preparation

Converts endpoint data into ShareGPT (training) and OpenAI messages (holdout) JSONL with stratified per-endpoint splits. The generated system prompt is parameterized via `--org-name` and defaults to `acme`.

### `endpoint_runner.py` — per-endpoint automation

Discovers endpoint directories, generates per-endpoint configs with auto-sized training parameters, and runs `cycle.py` for each. See `--help` for options.

### `emit_synth_status.py` — reposynth handoff

Emits `synth_status.yaml` from an eval JSON for the reposynth feedback loop. Called automatically by `cycle.py` step 5b. **Note:** the `format_health` section contains Go-specific heuristics — results are meaningless for non-Go outputs.

## See also

- [Architecture](docs/architecture.md) — detailed design notes on the pipeline, training approach, and evaluation.
- [Troubleshooting](docs/troubleshooting.md) — common failures and how to fix them.
- [Data Preparation](docs/data-preparation.md) — end-to-end guide from raw endpoint data to training JSONL.
- [DPO Fine-Tuning](docs/dpo.md) — when and how to use Direct Preference Optimization.
