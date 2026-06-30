# Lean Speculative Decoding — Bundle

This is a self-contained package for running Lean 4 tactic generation and
verification using fine-tuned Qwen2.5-Coder models on Mac (Apple Silicon).

## What's in this bundle

```
lean-speculative-bundle/
  README.md                     this file
  RUNBOOK.md                    step-by-step setup and execution guide
  FULL_EVAL.md                  full evaluation protocol and pass gates
  scripts/
    lean_eval.py                evaluation runner (generate + verify)
    lean_verify.py              Lean 4 safety checker and compiler wrapper
    make_report.py              report generator from eval summaries
  data/
    lean_stat/
      test.jsonl                1,866 held-out test records (state → tactic)
    sample/
      fixture_5.jsonl           5-record sanity fixture
      test_20.jsonl             20-record quick-smoke subset
  eval/
    lean_harness/
      Fixture.lean              5 known theorems for harness validation
      fixture_5.jsonl           matching fixture records
  fused-7b-lean/                fine-tuned 7B MLX model (fuse on Mac first)
  fused-0.5b-lean/              fine-tuned 0.5B draft MLX model (fuse on Mac)
  reports/                      eval output lands here after running lean_eval
```

## Quick start

**Prerequisites:** Mac with Apple Silicon, Python 3.10+, internet access.

```bash
# 1. Install mlx-lm
pip install "mlx-lm>=0.21.0"

# 2. Install Lean 4 via elan
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
echo 'source ~/.elan/env' >> ~/.zshrc   # add to shell profile
source ~/.elan/env

# 3. Copy MLX adapters from the training machine and fuse
#    (see RUNBOOK.md for the full rsync + fuse commands)

# 4. Quick smoke test (5 records, skip Lean)
python3 scripts/lean_eval.py \
  --model fused-7b-lean \
  --test data/sample/fixture_5.jsonl \
  --limit 5 --skip-lean \
  --output reports/lean_eval/smoke

# 5. Full run — see RUNBOOK.md
```

## Models

| Model | Role | Base |
|-------|------|------|
| fused-7b-lean | Target (fine-tuned) | Qwen2.5-Coder-7B-Instruct |
| fused-0.5b-lean | Draft (fine-tuned) | Qwen2.5-Coder-0.5B-Instruct |

Both were trained on `liminho123/lean4-stat-learning-theory-novel` for 2000
steps with LoRA rank 32, rsLoRA scaling, response-only loss masking.

## Background

Training used NVIDIA GB10 (121 GB VRAM) + Unsloth + PEFT. The PEFT adapters
were converted to MLX format with `peft_to_mlx.py` (in the source repo). This
bundle contains only the fused MLX models and evaluation code — no CUDA or
training dependencies required on Mac.

See RUNBOOK.md for complete setup and FULL_EVAL.md for the evaluation protocol.
