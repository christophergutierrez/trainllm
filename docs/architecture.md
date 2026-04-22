# Architecture

This document describes the design of the trainLLM pipeline in detail: how components relate, why key decisions were made, and what to expect at each stage.

## Overview

trainLLM is a three-script pipeline glued together by a config file and an orchestrator:

```
config.yaml
    │
    ├── train.py    ← Unsloth QLoRA fine-tuning
    ├── eval.py     ← vLLM-based holdout evaluation
    └── cycle.py    ← orchestrator (backup → train → serve → eval → report)
```

`_config.py` is a shared module that loads `config.yaml` and resolves all paths. Every script imports it, so there is a single source of truth for model names, paths, and hyperparameters.

---

## Config loading (`_config.py`)

`_config.py` is intentionally not a class — it exports a single `load()` function that returns a `SimpleNamespace`. This keeps imports simple (`cfg = _config.load()`) and avoids any module-level side effects before the config file is opened.

Paths are derived rather than stored: `lora_dir`, `final_dir`, `data_dir`, `evals_dir`, and `logs_dir` are all computed from `paths.base_dir` and `adapter_name`. This means renaming an adapter (changing `adapter_name` in config) automatically redirects all paths. Two adapters can coexist by simply changing `adapter_name` before each run.

`train.py` runs inside the Unsloth Python environment (a separate venv), which means it cannot import `_config` via a normal package mechanism. Instead, every script prepends its own directory to `sys.path` before importing:

```python
sys.path.insert(0, str(Path(__file__).parent))
import _config
```

This is robust to being called from any working directory.

---

## Training (`train.py`)

### QLoRA via Unsloth

Training uses [Unsloth](https://github.com/unslothai/unsloth) for memory-efficient QLoRA (Quantized Low-Rank Adaptation). The base model is loaded in 4-bit NF4 quantization, then a small set of trainable LoRA matrices is attached to the attention and MLP projection layers.

**Why QLoRA?** A 14B parameter model in full BF16 requires ~28 GB of VRAM just for weights. 4-bit quantization brings this to ~7 GB, leaving room for activations, gradients, and the LoRA adapter pages during training.

**Target modules:**

```
q_proj, k_proj, v_proj, o_proj   ← attention projections
gate_proj, up_proj, down_proj     ← MLP (SwiGLU) projections
```

These cover all learned weight matrices that matter for style adaptation. Embedding layers are left frozen — the base model's vocabulary coverage is kept intact.

**LoRA hyperparameters** (defaults from config):

| Parameter | Default | Effect |
|-----------|---------|--------|
| `lora_rank` | 16 | Capacity of the adapter; 16 is a reasonable starting point for domain adaptation |
| `lora_alpha` | 32 | Effective learning rate scaling: `alpha/rank = 2` is a common target |
| `lora_dropout` | 0 | Disabled — small ranks rarely need regularization from dropout |

Increasing rank increases adapter size and training time roughly linearly. Rank 32 or 64 makes sense if scores plateau and the base model capacity is not the bottleneck.

### Chat template and data format

Training data must be in **ShareGPT format**:

```json
{"conversations": [
  {"from": "human", "value": "..."},
  {"from": "gpt",   "value": "..."}
]}
```

Unsloth's `standardize_sharegpt` converts this to internal conversation objects; `get_chat_template` then applies the model-specific chat template (e.g., `<|im_start|>` tokens for Qwen, `<|begin_of_text|>` for Llama-3). The `chat_template` field in config must match the base model family, or the model will be trained on incorrectly formatted text and produce garbage outputs despite a converging loss.

`cycle.py` validates the ShareGPT structure of the training file before spending GPU time. It checks the first three records for the required `conversations → [from, value]` shape and fails fast with a descriptive error.

### Training dynamics

The default schedule is:

- **Batch size:** 2 per device × 4 gradient accumulation steps = 8 effective batch size
- **Warmup:** 50 steps (linear LR ramp from 0)
- **LR schedule:** cosine decay after warmup
- **Max steps:** 2000 (configurable)

**Rule of thumb for max_steps:** aim for approximately 5–10 epochs over your dataset. With ~2000 training records and effective batch size 8, one epoch ≈ 250 steps, so 2000 steps ≈ 8 epochs. If you have 500 records, 2000 steps is ~32 epochs — likely overfit. Scale accordingly.

**Loss interpretation:**
- Loss converging to ~0.5–0.8 by the end is typical for instruction fine-tuning.
- Loss > 1.0 at the end of training: undertrained — increase `max_steps`.
- Loss < 0.15: overfit — reduce `max_steps` or add regularization.
- Loss not decreasing at all: check `chat_template` matches the model; check data format.

The adapter is saved to `lora/<adapter_name>/` as checkpoints every `save_steps` steps (keeping the last `save_total_limit`), with a final save to `lora/<adapter_name>/final/`.

### Optimizer note

`adamw_8bit` (bitsandbytes) is disabled on CUDA 13 due to a compatibility issue. `adamw_torch` (standard PyTorch) is used instead. This has a small memory overhead (~10–15%) but is otherwise equivalent.

---

## Orchestration (`cycle.py`)

`cycle.py` glues the pipeline together and handles all the operational concerns that training and evaluation scripts don't know about: backups, process lifecycle, timeouts, and cross-run comparison.

### Step sequence

```
1. Backup      Copy lora/<adapter>/final → lora/<adapter>/final-v<date>
2. Stop vLLM   SIGTERM → wait 3s → SIGKILL if still running
3. Train       Run train.py under Unsloth Python; stream and parse output
4. Start vLLM  Launch vllm serve with final/ + saved checkpoints; poll /v1/models until ready
5. Eval        Run eval.py for each candidate (final/ + saved checkpoints)
5a. Best-checkpoint  Score each checkpoint; promote highest-scoring to final/ (atomic: copy to .tmp_promote → rename)
5b. Synth status     Emit synth_status.yaml from winning eval for reposynth handoff
6. Report      Compare scores, print diagnosis, log paths to eval reports
```

Steps 1–3 are skipped with `--skip-train`. Step 4 is skipped with `--skip-serve`. Steps 5–6 are skipped with `--skip-eval`. The base model eval (second half of step 5) is skipped with `--skip-base-eval`.

### WatchdogProcess

Training and eval are wrapped in `WatchdogProcess`, a lightweight subprocess manager that enforces two independent timeouts:

**Silence timeout** — kills the process if no stdout line is received within N seconds. Used for training, because Unsloth emits a log line every 10 steps; N minutes of silence means a GPU deadlock.

**Wall timeout** — kills the process after N total seconds regardless of output. Used for eval, where a single stuck vLLM request can block indefinitely without being silent.

Both timers run in a background watchdog thread. The main thread streams stdout line by line (so output appears in real time in the log), and calls an optional `line_callback` per line. This callback is used during training to parse loss values from the HuggingFace trainer output format:

```
{'loss': 0.4231, 'grad_norm': 1.234, 'epoch': 0.45}
```

The parsed loss curve is used to detect early warning signs (loss > 1.5 past epoch 0.2) and to emit a final convergence summary.

PID files are written to `logs/` at process start and removed on clean exit, giving an external handle if the cycle needs to be interrupted.

### vLLM startup

vLLM is launched with:

```bash
vllm serve <base_model>
  --dtype bfloat16
  --gpu-memory-utilization 0.85
  --enforce-eager               # Blackwell: prevents torch.compile hang
  --enable-lora
  --lora-modules <adapter_name>=<final_dir>
  --port <port>
```

The `--enforce-eager` flag is required on Blackwell (GB10) architecture. Without it, vLLM enters `torch.compile` during model load and hangs indefinitely.

`--gpu-memory-utilization 0.85` leaves ~15% headroom. vLLM pre-allocates a KV cache from this budget; the remaining space is needed for the LoRA adapter pages that are loaded on demand when a request specifies the adapter.

After launching, `cycle.py` polls `GET /v1/models` every `vllm_poll_interval` seconds until the server responds, up to `vllm_startup_timeout` seconds. It checks that the adapter name appears in the model list — if it doesn't, the eval will silently test the base model and appear to show no fine-tuning effect.

### Comparison report

After both evals complete, `step_report` loads their JSON outputs and produces a structured summary:

- **Score bar** — visual 0–1 bar for fine-tuned vs base.
- **Delta** — fine-tuned minus base; ≥0.05 is considered meaningful.
- **Band breakdown** — counts per quality tier (Excellent/Good/Partial/Poor/Error).
- **Convention breakdown** — per-tag averages if `conventions_tested` is populated in the holdout.
- **Run-over-run comparison** — compares the current fine-tuned eval against the previous run (by mtime) to surface regressions and improvements at the example level.
- **Diagnosis** — a short text block interpreting the numbers and suggesting next actions.

---

## Evaluation (`eval.py`)

### How it works

`eval.py` queries the vLLM server (OpenAI-compatible API) with the user turn from each holdout record, then scores the generated response against the reference answer using **token-level sequence similarity** (`difflib.SequenceMatcher`).

This metric rewards responses that share token sequences with the reference. It is fast, deterministic, and requires no secondary model. Its main weakness is that semantically equivalent outputs with different wording score poorly — treat it as a lower bound on quality, not a definitive measure.

### Holdout format

The holdout file uses OpenAI message format (not ShareGPT) because it is read by `eval.py` with the standard `openai` client, not Unsloth:

```json
{
  "messages": [
    {"role": "user",      "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "id": "...",
  "label": "...",
  "conventions_tested": ["tag-a", "tag-b"]
}
```

The eval strips the `assistant` turn before sending to vLLM, then compares the generated response to it.

### Scoring and bands

Similarity is computed after stripping markdown code fences (` ``` `) from both sides, so fence presence or absence does not affect the score.

| Band | Range |
|------|-------|
| EXCELLENT | ≥ 0.8 |
| GOOD | ≥ 0.6 |
| PARTIAL | ≥ 0.4 |
| POOR | < 0.4 |

`length_ratio` (generated length / expected length) is recorded per example as a diagnostic. A ratio far from 1.0 (very short or very long outputs) often correlates with format failures.

### Convention breakdown

If holdout records include `conventions_tested` tags, `eval.py` groups scores by tag and computes per-tag averages. This is the primary mechanism for diagnosing *which* behaviors the model has not learned, rather than just the overall average. Tags are freeform strings — define them to match the structure of your training data.

### Output files

Two files are written per run, named `<timestamp>_<safe_model_name>`:

- `.json` — full raw data: all prompts, expected outputs, generated outputs, scores, metadata.
- `.md` — human-readable report: score table, convention breakdown, per-example results sorted worst-first, full prompt/expected/generated for failing cases, and suggested next steps.

---

## Data flow diagram

```
config.yaml
    │
    ▼
_config.py ──────────────────────────────────────────┐
    │                                                 │
    ▼                                                 ▼
train.py                                           eval.py
    │                                                 │
    │  reads: data/training.jsonl (ShareGPT)          │  reads: data/holdout.jsonl (messages)
    │  via: Unsloth Python venv                       │  queries: vLLM HTTP API
    │                                                 │
    ▼                                                 ▼
lora/<adapter>/final/              evals/<timestamp>_<model>.{md,json}
    │
    ▼
vllm serve --lora-modules <adapter>=lora/<adapter>/final/
    │
    └──► eval.py queries via OpenAI client
```

`cycle.py` drives this sequence top to bottom, managing process lifecycles, logging, and cross-run comparison.

---

## Adapter lifecycle

```
Run N:
  lora/<adapter>/final/          ← trained by run N-1
      → copied to final-v<date>  ← backup before run N
      → overwritten by run N     ← new adapter

Run N+1:
  lora/<adapter>/final/          ← trained by run N
      → copied to final-v<date>  ← backup (suffix -1 if same day)
      ...
```

Backups are never automatically pruned. Remove old `final-v*` directories manually when disk space is needed. The `save_total_limit` setting in config controls checkpoint retention *during* training (Hugging Face saves intermediate checkpoints); it has no effect on the `final-v*` backups.

---

## Multi-adapter usage

To fine-tune multiple models or train multiple adapters simultaneously:

1. Copy the entire `~/trainLLM` directory to a new location (e.g., `~/trainLLM-codegen`).
2. Edit `config.yaml` in the new directory: change `model`, `adapter_name`, and data paths.
3. Run `cycle.py` from the new directory.

Because all paths derive from `config.yaml` and scripts locate `_config.py` via `__file__`, copies are fully independent. The vLLM port (`vllm.port`) must differ if both pipelines need to serve simultaneously.

---

## Extending the pipeline

**Different base model:** change `model` and `chat_template` in `config.yaml`. No code changes needed.

**Different data:** place new JSONL at the configured path, or use `--version` / `--holdout` flags. Training data must be ShareGPT format; holdout must be messages format.

**Different hyperparameters:** edit `training.*` in `config.yaml`.

**Custom eval metric:** replace the `similarity()` function in `eval.py`. The rest of the scoring, reporting, and aggregation logic is metric-agnostic.

**Adding a pre/post-processing step:** add a function to `cycle.py` and call it in `main()` between the existing steps. The `WatchdogProcess` class handles any subprocess that writes to stdout.
