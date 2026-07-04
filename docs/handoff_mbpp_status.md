# MBPP Handoff Status

Date: 2026-07-04

## Goal

Build a speculative decoding demo that shows:

1. Fine-tuning improves code correctness.
2. A smaller draft model speeds up inference.

Current approach uses MBPP-style function-writing data because Magicoder did not match the eval task well enough.

## Current Data

Prepared by `prepare_mbpp_data.py`:

- Train: `data/mbpp/train.jsonl`
  - Source: `nlile/mbpp` train split
  - 374 records
- Holdout: `data/mbpp/holdout.jsonl`
  - Source: `nlile/mbpp` validation split
  - 90 records
- Final eval: `data/mbpp/mbpp_plus_test.jsonl`
  - Source: `evalplus/mbppplus`
  - 378 records

Manifest version: `mbpp-function-tests-v1`

## Important Fixes Already Made

- `mbpp_eval.py` verifies code by writing solution+tests to a temp `.py` file, not `python -c`, because MBPP+ tests can exceed OS argv length.
- `train.py` supports `--probe-kind mbpp`.
- `train.py` uses explicit holdout data, early stopping, and best-checkpoint loading.
- `run_mbpp_pipeline.sh` is gated to avoid wasting 7B training time.

## Pipeline

Run:

```bash
tmux new-session -d -s mbpp "cd /home/chris/git_home/trainllm && bash run_mbpp_pipeline.sh 2>&1 | tee /tmp/mbpp.log"
```

Monitor:

```bash
tail -f /tmp/mbpp.log
```

Gates in `run_mbpp_pipeline.sh`:

- Base 0.5B smoke on first 50 MBPP+ tasks must be >= 10%.
- Trained 0.5B must beat base 0.5B by at least 2 percentage points.
- Base 7B smoke must be >= 10%.
- Base 7B full eval stops 7B training if already > 85%.

## Latest Result

The current/last MBPP run reached the 0.5B gate and failed it:

- Base 0.5B MBPP+: `39.4%`
- Trained 0.5B MBPP+: `36.5%` (`138/378`)
- Delta: `-2.9%`

This means the gate should stop before any 7B work. Do not train 7B from this run.

## Interpretation

This is now a cleaner negative result than the earlier Magicoder result:

- Data is same task family.
- Eval harness is fixed.
- Training uses holdout early stopping and best checkpoint.
- 0.5B still regressed.

Likely causes:

- MBPP train is small: only 374 training examples.
- The base `Qwen2.5-Coder-0.5B-Instruct` already has strong benchmark priors.
- SFT on a tiny public-test prompt format may overfit style/tests rather than improve MBPP+ hidden-test robustness.

## Recommended Next Decisions

Do not proceed to 7B training from the current 0.5B result.

Reasonable next experiments:

1. **Try a weaker non-coder base model** for the quality-improvement story.
   - Example target family: `Qwen/Qwen2.5-0.5B-Instruct` and `Qwen/Qwen2.5-7B-Instruct`.
   - Hypothesis: MBPP SFT may improve a general model more clearly than an already-code-specialized model.

2. **Use more training data with tests** before retrying coder models.
   - Combine MBPP train/validation plus generated/test-backed Python function tasks, but keep MBPP+ as holdout.
   - Avoid using MBPP+ eval tasks as training.

3. **Separate the two demo claims.**
   - For quality: use a model/data pair that actually improves.
   - For speed: use the best available small draft model and target model, even if draft was not improved by this exact SFT.

## Current Files Added/Changed For MBPP

- `prepare_mbpp_data.py`
- `mbpp_eval.py`
- `config.mbpp-0.5b.yaml`
- `config.mbpp-7b.yaml`
- `run_mbpp_pipeline.sh`
- `docs/mbpp_plan.md`
- `docs/handoff_mbpp_status.md`
- `train.py` changed for MBPP probe and best-checkpoint/holdout handling.
- `tests/test_config.py` updated to include MBPP configs.

## Verification

Most recent full test suite before the MBPP run:

```text
205 passed, 2 warnings
```

