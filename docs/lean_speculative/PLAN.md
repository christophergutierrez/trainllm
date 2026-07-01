# Lean Speculative Decoding Plan

This plan builds an end-to-end example for training Lean 4 tactic models on the
GB10 machine, baking them into MLX-compatible fused models, and running them on
either the GB10 or a Mac laptop.

The target result is a reproducible workflow:

1. Prepare Lean state/tactic instruction data.
2. Train a 7B target adapter and a 0.5B draft adapter on GB10.
3. Convert PEFT adapters to MLX adapter format.
4. Fuse both adapters into standalone MLX model directories.
5. Benchmark target-only generation against speculative decoding.
6. Verify generated tactics with Lean, not manual inspection.
7. Package only scripts, docs, configs, and final fused model directories.

## Non-Goals

- Do not check model caches, checkpoints, fused weights, or HF downloads into git.
- Do not require Mac users to install CUDA, Unsloth, or training dependencies.
- Do not claim speculative decoding improves correctness. It should preserve the
  target model distribution and improve speed when the draft is accepted often
  enough.
- Do not treat a text match with the dataset tactic as the main correctness
  metric. Lean compiler acceptance is the main correctness gate.

## Assumptions

- Training happens on the GB10 machine using the existing `trainllm` Unsloth path.
- Mac execution uses `mlx-lm`.
- Raw `trainllm` adapters are PEFT adapters and must be converted before use with
  `mlx_lm.fuse`.
- Base models:
  - Target: `Qwen/Qwen2.5-Coder-7B-Instruct`
  - Draft: `Qwen/Qwen2.5-Coder-0.5B-Instruct`
- Dataset starting point:
  - `liminho123/lean4-stat-learning-theory-novel`

## Phase 0: Repo And Environment Audit

Goal: confirm the current repo can support the workflow before adding feature
code.

### Milestone 0.1: Training CLI Audit

Required implementation:

- Inspect `train.py`.
- Decide whether to add argparse or preserve env vars plus a wrapper script.
- Required runtime knobs:
  - `--base-model`
  - `--train-data`
  - `--output-dir`
  - `--max-steps`
  - optional `--adapter-name`

Tests before moving on:

- `uv run train.py --help` exits 0.
- Help output documents all required runtime knobs.
- A dry-run or config-print mode shows the requested base model and output dir.
- Existing config-driven training remains backward compatible.

### Milestone 0.2: MLX Capability Audit

Required implementation:

- Confirm installed or installable `mlx-lm` exposes:
  - `mlx_lm.generate --draft-model`
  - `mlx_lm.generate --num-draft-tokens`
  - `mlx_lm.fuse --adapter-path`
- Confirm `peft_to_mlx.py` conversion format matches MLX expectations.

Tests before moving on:

- `mlx_lm.generate --help` contains `--draft-model`.
- `mlx_lm.fuse --help` contains `--adapter-path`.
- `uv run peft_to_mlx.py --help` exits 0.
- `uv run pytest tests/test_peft_to_mlx.py` passes.

### Milestone 0.3: Git Hygiene Audit

Required implementation:

- Ensure generated artifacts stay ignored:
  - `adapters/`
  - `fused-*`
  - `models/`
  - `*.safetensors` except intentional tiny test fixtures
  - `.cache/`
  - `__pycache__/`

Tests before moving on:

- `git status --ignored --short` shows generated artifacts ignored.
- No large cache/model files are tracked.
- New docs/scripts are tracked or intentionally staged by the operator.

## Phase 1: Lean Data Pipeline

Goal: create deterministic train/valid/test data in the chat JSONL format the
training pipeline expects.

### Milestone 1.1: Data Preparation Script

Required implementation:

- Add `prepare_lean_data.py`.
- Download `liminho123/lean4-stat-learning-theory-novel`.
- Produce:
  - `data/lean_stat/train.jsonl`
  - `data/lean_stat/valid.jsonl`
  - `data/lean_stat/test.jsonl`
- Each record must be a single-line JSON object:

```json
{"conversations":[{"from":"human","value":"Given the Lean 4 state:\n...\nProvide the next tactical step."},{"from":"gpt","value":"..."}]}
```

Tests before moving on:

- Script runs from a clean checkout with `uv run prepare_lean_data.py`.
- Output JSONL files exist and are non-empty.
- Every line parses as JSON.
- Every record has exactly one user message and one assistant message.
- No empty states or empty tactics.
- Split is deterministic for a fixed seed.

### Milestone 1.2: Data Quality Filters

Required implementation:

- Add filtering or flags for tactics containing:
  - `sorry`
  - `admit`
  - `by_cases` only if later found to mask bad outputs
- Preserve a reject report with counts and examples.
- Add a deterministic cleanup pass that can normalize whitespace, drop malformed
  records, and dedupe exact `(state, tactic)` pairs across the cleaned splits.
  The cleanup output should be a separate checked-in script and a reproducible
  output directory, not a manual one-off.

Tests before moving on:

- Reject report is written.
- Filtered records contain no forbidden tokens when strict mode is enabled.
- A small fixture test covers valid, invalid, and edge-case records.

### Milestone 1.3: Prompt Rendering Check

Required implementation:

- Add a script or test that renders sample conversations through the Qwen chat
  template used by training.

Tests before moving on:

- 10 samples render without tokenizer errors.
- Rendered prompt contains the Lean state.
- Rendered prompt contains the target tactic only in the assistant segment.
- Response-only loss masking still masks user tokens.

## Phase 2: Training Pipeline

Goal: train PEFT adapters for both target and draft models on GB10.

### Milestone 2.1: 0.5B Smoke Adapter

Required implementation:

- Train a short 0.5B run on a tiny slice.
- Write outputs under `adapters/0.5b-lean-smoke/`.

Tests before moving on:

- Training starts with `Qwen/Qwen2.5-Coder-0.5B-Instruct`.
- Output directory contains PEFT adapter files.
- Loss is logged.
- Final adapter loads with Transformers/PEFT.
- Resume from checkpoint works if interrupted.

### Milestone 2.2: 7B Smoke Adapter

Required implementation:

- Train a short 7B run on a tiny slice.
- Write outputs under `adapters/7b-lean-smoke/`.

Tests before moving on:

- Training starts with `Qwen/Qwen2.5-Coder-7B-Instruct`.
- Output directory contains PEFT adapter files.
- Loss is logged.
- Final adapter loads with Transformers/PEFT.
- GB10 memory settings are documented if they differ from defaults.

### Milestone 2.3: Full 0.5B Adapter

Required implementation:

- Train full draft adapter.
- Write outputs under `adapters/0.5b-lean/`.

Tests before moving on:

- Final adapter exists.
- Convergence summary exists.
- Loss trend is sane.
- A small generation smoke test produces Lean-like tactic text.

### Milestone 2.4: Full 7B Adapter

Required implementation:

- Train full target adapter.
- Write outputs under `adapters/7b-lean/`.

Tests before moving on:

- Final adapter exists.
- Convergence summary exists.
- Loss trend is sane.
- A small generation smoke test produces Lean-like tactic text.

Preferred next-step target, if the machine can support it comfortably:

- Add a 14B configuration based on `Qwen/Qwen2.5-Coder-14B-Instruct`.
- Treat 7B as the fallback baseline and 14B as the stronger capacity probe.
- Keep the same deterministic data preparation and cleanup path for both.

## Phase 3: MLX Conversion And Fusion

Goal: convert GB10-trained PEFT adapters to MLX adapters, then fuse them into
standalone MLX model directories.

### Milestone 3.1: Convert 0.5B Adapter

Required implementation:

```bash
uv run peft_to_mlx.py --in adapters/0.5b-lean/final --out adapters/0.5b-lean-mlx
```

Tests before moving on:

- `adapters/0.5b-lean-mlx/adapters.safetensors` exists.
- `adapters/0.5b-lean-mlx/adapter_config.json` exists.
- MLX generation with `--adapter-path` runs on one prompt.

### Milestone 3.2: Convert 7B Adapter

Required implementation:

```bash
uv run peft_to_mlx.py --in adapters/7b-lean/final --out adapters/7b-lean-mlx
```

Tests before moving on:

- `adapters/7b-lean-mlx/adapters.safetensors` exists.
- `adapters/7b-lean-mlx/adapter_config.json` exists.
- MLX generation with `--adapter-path` runs on one prompt.

### Milestone 3.3: Fuse 0.5B Model

Required implementation:

```bash
mlx_lm.fuse \
  --model Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --adapter-path adapters/0.5b-lean-mlx \
  --save-path fused-0.5b-lean
```

Tests before moving on:

- `fused-0.5b-lean/` exists.
- `mlx_lm.generate --model fused-0.5b-lean --prompt ...` exits 0.
- Output is non-empty.

### Milestone 3.4: Fuse 7B Model

Required implementation:

```bash
mlx_lm.fuse \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --adapter-path adapters/7b-lean-mlx \
  --save-path fused-7b-lean
```

Tests before moving on:

- `fused-7b-lean/` exists.
- `mlx_lm.generate --model fused-7b-lean --prompt ...` exits 0.
- Output is non-empty.
- Fused 7B output roughly matches unfused MLX adapter output under deterministic
  settings.

## Phase 4: Lean Verification Harness

Goal: make correctness automatic.

### Milestone 4.1: Lean Project Harness

Required implementation:

- Add a small Lean project fixture or document how to create one.
- Add a script that can insert a generated tactic into a theorem context and
  invoke Lean.

Tests before moving on:

- A known-valid tactic passes.
- A known-invalid tactic fails.
- Missing Lean executable produces a clear error.
- The harness captures compiler stdout/stderr.

### Milestone 4.2: Tactic Safety Checks

Required implementation:

- Reject generated tactics containing forbidden tokens:
  - `sorry`
  - `admit`
  - `by sorry`
  - placeholder markers

Tests before moving on:

- Safety check rejects known bad examples.
- Safety check accepts known good examples.
- Compiler pass is not counted when safety check fails.

### Milestone 4.3: Evaluation Runner

Required implementation:

- Add a runner that:
  - reads `data/lean_stat/test.jsonl`
  - generates one tactic per record
  - verifies each tactic with Lean
  - records latency and tokens/sec
  - writes JSONL predictions and JSON summary

Tests before moving on:

- Runner works on a 5-record fixture.
- Output JSONL has prompt, expected tactic, generated tactic, compiler result,
  elapsed seconds, and tokens/sec.
- Summary includes compile pass rate and error counts.

## Phase 5: Benchmark Matrix

Goal: compare correctness and speed across base, fine-tuned, and speculative
execution.

### Milestone 5.1: Baseline Runs

Required implementation:

- Run:
  - base 7B
  - fused 7B
  - fused 0.5B

Tests before moving on:

- All runs use the same test split.
- All runs use deterministic decoding.
- All runs produce prediction JSONL and summary JSON.
- Compiler pass rate is reported for each run.

### Milestone 5.2: Speculative Run

Required implementation:

```bash
mlx_lm.generate \
  --model fused-7b-lean \
  --draft-model fused-0.5b-lean \
  --num-draft-tokens 5 \
  --prompt "..."
```

Tests before moving on:

- Speculative generation exits 0.
- Tokens/sec is captured.
- Compiler pass rate is reported.
- Target-only and speculative settings are recorded.
- Output quality is compared against target-only, not assumed.

### Milestone 5.3: Report

Required implementation:

- Generate a Markdown report with:
  - dataset sizes
  - model names and revisions
  - training settings
  - compile pass rate
  - mean/median latency
  - tokens/sec
  - representative failures

Tests before moving on:

- Report regenerates from JSON summaries.
- Report contains no absolute `/home/chris/...` paths.
- Report clearly separates correctness claims from speed claims.

## Phase 6: Handoff Package

Goal: create a Mac/GB10 runnable package without training junk.

### Milestone 6.1: Bundle Layout

Required package layout:

```text
lean-speculative-bundle/
  README.md
  RUNBOOK.md
  FULL_EVAL.md
  scripts/
  data/sample/
  fused-7b-lean/
  fused-0.5b-lean/
  reports/
```

Tests before moving on:

- Package contains no checkpoints.
- Package contains no HF cache directories.
- Package contains no `__pycache__`.
- Package contains no absolute `/home/chris/...` paths.

### Milestone 6.2: Cold-Agent Validation

Required implementation:

- Start a cheap fresh subagent with only the bundle path.
- Ask it to produce a run plan and identify blockers.
- Fix real blockers.
- Repeat up to 10 loops.

Tests before moving on:

- Cold agent can explain how to run target-only and speculative inference.
- Cold agent can explain how to run Lean verification.
- No bundle-internal blockers remain.
