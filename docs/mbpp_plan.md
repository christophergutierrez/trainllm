# MBPP / MBPP+ Training Plan

Goal: prove whether SFT improves code correctness before spending time on large
model training, then use the improved 0.5B as the draft model for speed tests.

## Data

- Training: `nlile/mbpp` train split
- Training holdout: `nlile/mbpp` validation split
- Final quality eval: `evalplus/mbppplus` test split

The prepared examples include the task text and public tests in the prompt, and
the reference solution as the assistant response. Final evaluation uses MBPP+
tests so the held-out metric is stricter than the training examples.

## Gates

1. Smoke-evaluate base 0.5B on 50 MBPP+ tasks.
   Stop if pass@1 is below 10%, because the eval harness or prompt is broken.
2. Full-evaluate base 0.5B.
3. Train 0.5B with validation every 25 steps, early stopping, and best-checkpoint loading.
4. Full-evaluate trained 0.5B.
   Stop unless trained 0.5B beats base 0.5B by at least 2 percentage points.
5. Smoke-evaluate base 7B on 50 MBPP+ tasks.
   Stop if pass@1 is below 10%.
6. Full-evaluate base 7B.
   Stop if pass@1 exceeds 85%, because quality headroom is too small.
7. Train and evaluate 7B only after all previous gates pass.

Run:

```bash
tmux new-session -d -s mbpp "bash run_mbpp_pipeline.sh 2>&1 | tee /tmp/mbpp.log"
```
