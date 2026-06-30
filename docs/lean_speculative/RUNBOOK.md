# Lean Speculative Decoding — Runbook

Step-by-step instructions for running the full evaluation on a Mac with Apple
Silicon. Training ran on the GB10 (Linux, CUDA). Only the Mac steps are
documented here.

## Prerequisites

### 1. Python with MLX

```bash
python3 -m pip install mlx-lm
```

Verify speculative decoding is available:

```bash
python3 -m mlx_lm.generate --help | grep draft
# should show: --draft-model and --num-draft-tokens
```

### 2. Lean 4 via elan

```bash
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
source ~/.elan/env
lean --version
```

### 3. Clone the repo

```bash
git clone <repo-url> trainllm
cd trainllm
```

## Step 1: Copy MLX adapters from GB10

The PEFT adapters were converted on GB10. Copy the MLX-format adapter
directories to the Mac:

```bash
rsync -av gb10:~/git_home/trainllm/adapters/0.5b-lean-mlx/ adapters/0.5b-lean-mlx/
rsync -av gb10:~/git_home/trainllm/adapters/7b-lean-mlx/   adapters/7b-lean-mlx/
```

Verify both contain `adapters.safetensors` and `adapter_config.json`.

## Step 2: Fuse adapters into standalone models (Phase 3.3–3.4)

```bash
python3 -m mlx_lm.fuse \
  --model Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --adapter-path adapters/0.5b-lean-mlx \
  --save-path fused-0.5b-lean

python3 -m mlx_lm.fuse \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --adapter-path adapters/7b-lean-mlx \
  --save-path fused-7b-lean
```

Smoke-check both:

```bash
python3 -m mlx_lm.generate \
  --model fused-0.5b-lean \
  --prompt "Given the Lean 4 state:\nn : Nat\n⊢ n + 0 = n\nProvide the next tactical step." \
  --max-tokens 32 --temp 0

python3 -m mlx_lm.generate \
  --model fused-7b-lean \
  --prompt "Given the Lean 4 state:\nn : Nat\n⊢ n + 0 = n\nProvide the next tactical step." \
  --max-tokens 32 --temp 0
```

## Step 3: Verify Lean harness (Phase 4)

Quick sanity check (no model needed):

```bash
python3 lean_verify.py --check "simp"        # → SAFE
python3 lean_verify.py --check "sorry"       # → FORBIDDEN

python3 lean_verify.py --file eval/lean_harness/Fixture.lean
# → {"passed": true, ...}
```

Full fixture test (5 records, requires fused model):

```bash
python3 lean_eval.py \
  --model fused-7b-lean \
  --test eval/lean_harness/fixture_5.jsonl \
  --limit 5 \
  --output reports/lean_eval/fixture/predictions.jsonl \
  --summary reports/lean_eval/fixture/summary.json
```

## Step 4: Baseline evaluation runs (Phase 5.1)

### Base 7B (untuned)

```bash
python3 lean_eval.py \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --test data/lean_stat/test.jsonl \
  --output reports/lean_eval/base-7b/predictions.jsonl \
  --summary reports/lean_eval/base-7b/summary.json
```

### Fused 7B target

```bash
python3 lean_eval.py \
  --model fused-7b-lean \
  --test data/lean_stat/test.jsonl \
  --output reports/lean_eval/target-7b/predictions.jsonl \
  --summary reports/lean_eval/target-7b/summary.json
```

### Fused 0.5B draft alone

```bash
python3 lean_eval.py \
  --model fused-0.5b-lean \
  --test data/lean_stat/test.jsonl \
  --output reports/lean_eval/draft-0.5b/predictions.jsonl \
  --summary reports/lean_eval/draft-0.5b/summary.json
```

## Step 5: Speculative run (Phase 5.2)

```bash
python3 lean_eval.py \
  --model fused-7b-lean \
  --draft-model fused-0.5b-lean \
  --num-draft-tokens 5 \
  --test data/lean_stat/test.jsonl \
  --output reports/lean_eval/speculative/predictions.jsonl \
  --summary reports/lean_eval/speculative/summary.json
```

## Step 6: Generate the report (Phase 5.3)

```bash
python3 make_report.py
# writes reports/REPORT.md
```

Review the report:

```bash
cat reports/REPORT.md
```

## Step 7: Assemble the handoff bundle (Phase 6.1)

```bash
python3 make_bundle.py --validate-only   # check first
python3 make_bundle.py                   # assemble
```

The bundle is written to `lean-speculative-bundle/`.

## Smoke run (quick sanity check for the whole pipeline)

Use `--limit 5` on all eval runs and skip Lean for fast iteration:

```bash
python3 lean_eval.py --model fused-7b-lean \
  --test data/lean_stat/test.jsonl --limit 5 --skip-lean \
  --output /tmp/smoke.jsonl --summary /tmp/smoke-summary.json

python3 lean_eval.py --model fused-7b-lean \
  --draft-model fused-0.5b-lean --num-draft-tokens 5 \
  --test data/lean_stat/test.jsonl --limit 5 --skip-lean \
  --output /tmp/smoke-spec.jsonl --summary /tmp/smoke-spec-summary.json
```

## Troubleshooting

**`mlx_lm.generate` crashes with "weight not found"**
The PEFT → MLX conversion may have produced misaligned keys. Rerun
`peft_to_mlx.py` and verify the `keys` field in `adapter_config.json` matches
the module names in the base model.

**Lean verification always returns `lean_ok: null`**
Lean is not in PATH or the goal state cannot be reconstructed as a standalone
`example`. Run `which lean` and `lean --version` to confirm Lean is installed.

**Speculative decoding is slower than target-only**
Expected when the draft acceptance rate is low. The 0.5B draft was trained on
the same data distribution, so acceptance should be reasonable. If tokens/sec
is lower, check that both `--model` and `--draft-model` are passed correctly.

**`mlx_lm.fuse` reports missing keys**
Ensure `adapter_config.json` in the MLX adapter directory contains the correct
`keys` field (module names without `base_model.model.` prefix). The
`peft_to_mlx.py` script sets these correctly for Qwen2.5 models.
