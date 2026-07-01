# Lean Tactic Evaluation Summary

This note records the measured evaluation results we have so far for the Lean
tactic prediction project.

Scope:

- Local model runs are measured from the saved `results/*/summary.json` files.
- The Sonnet entry is a partial sample and is marked with `*`.
- No speculative decoding run has a completed measured result yet, so it is not
  included as a benchmark row.

## Measured Results

| Run | Model | n evaluated | Lean pass | Lean fail | Lean skip | Pass rate | Mean sec / record | Mean tokens / sec | Total tokens | Total elapsed | Approx cost |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Base 0.5B | Qwen2.5-Coder-0.5B-Instruct | 1215 | 0 | 1213 | 2 | 0.00% | 2.019 | 124.4 | 305,070 | 2453.1 s | NA |
| Target 0.5B | 0.5B-lean adapter | 1215 | 25 | 1188 | 2 | 2.06% | 0.281 | 86.2 | 29,484 | 341.9 s | NA |
| Base 7B | Qwen2.5-Coder-7B-Instruct | 1215 | 4 | 1206 | 5 | 0.33% | 1.780 | 11.0 | 23,732 | 2162.1 s | NA |
| Target 7B | 7B-lean adapter, reverified on current harness | 1215 | 45 | 1168 | 2 | 3.71% | 2.144 | 10.7 | 27,995 | 2604.8 s | NA |
| Sonnet sample* | Partial Sonnet sample, 305 records | 305 | 18 | 284 | 3 | 5.96% | 3.370 | n/a | n/a | n/a | sample only; full run roughly $19-$24* |

* Sonnet cost is a napkin estimate for a full 1215-record no-context run, using
current Sonnet 4.6 pricing and the same rough token mix we observed in the
sample workflow. It is not a measured cost.

Pass rate is Lean pass / (Lean pass + Lean fail). For the Sonnet sample, the
three skips are excluded from the denominator. The Target 7B mean seconds per
record is recomputed from total elapsed so the row stays internally consistent.
The Target 7B Lean counts come from `results/target-7b-reverified/summary.json`;
its generation timing and token counts remain from the original run summary.

## Interpretation

The measured local results show a clear improvement from the base models to the
fine-tuned adapters:

- The 0.5B adapter moves from 0.00% Lean pass to 2.06%.
- The 7B adapter moves from 0.33% Lean pass to 3.71% on the current harness.
- The 7B adapter is the strongest completed local run we have.

The Sonnet sample is slightly better on Lean pass rate than the completed 7B
adapter, but it is only a 305-record partial sample. It should be treated as
representative evidence, not a final benchmark.

## Why This Matters

This is not a small benchmark flourish. If you count only focused analysis and
coordination time, I would estimate the work at roughly 2 to 3 hours, which is
about 1/4 to 1/3 of an 8-hour day. That is a meaningful block of engineering
effort even before you count the wait time for the model to run.

The practical difference is in dollars:

- A no-context Sonnet full run is roughly $19-$24.
- Adding the obvious training context lifts that to roughly $29-$37.
- If the same workflow were run with Opus instead, the no-context version is
  roughly $32-$40.
- With context, Opus rises to roughly $49-$61.

For a company, the useful budget unit is per employee per month. As of the
current Claude pricing page, Max starts at $100 per month per employee. That is
$1,000/month for 10 employees, $5,000/month for 50 employees, and $10,000/month
for 100 employees before any API usage or internal infrastructure.

If an employee burns through even a handful of benchmark-sized runs per month,
the monthly spend moves fast. Five Sonnet no-context runs is about $95-$120.
Five context-heavy Sonnet runs is about $145-$185. Ten context-heavy Opus runs
is about $490-$610. That is why "let them token max to get shit done" is not a
free productivity hack. It is a budget policy.

## Notes On The Sonnet Sample

The Sonnet row is based on 305 completed records out of a 1215-record planned
run. The run stopped early because the plan token budget was exhausted.

Because the run is incomplete, the full-run cost numbers above are napkin math,
not measurements. The row is still useful because it shows a real sample that
was expensive enough to matter and strong enough to compete with the trained 7B
adapter on the recovered subset.

## Data Sources

- `results/base-0.5b/summary.json`
- `results/target-0.5b/summary.json`
- `results/base-7b/summary.json`
- `results/target-7b/summary.json`
- `results/target-7b-reverified/summary.json`
- `results/sonnet-lean-eval-v4-recovered/summary.json`
