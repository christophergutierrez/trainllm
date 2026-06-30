# Full Evaluation Protocol

This document defines the full evaluation once the training, conversion, fusion,
and Lean verification harness are implemented.

## Evaluation Questions

The evaluation should answer:

1. Does fine-tuning improve Lean tactic correctness over the base 7B model?
2. Does the 0.5B draft model preserve target quality when used for speculative
   decoding?
3. Does speculative decoding improve generation speed on Mac/MLX?
4. What failures remain, and are they model failures, prompt failures, or Lean
   harness failures?

## Test Set

Use `data/lean_stat/test.jsonl`.

Requirements:

- The test split must be deterministic.
- Test records must not be used for training.
- The report must include record count.
- If a smaller smoke subset is used, label it clearly as smoke, not full eval.

## Models In Matrix

### Base 7B

Purpose: measure the pretrained baseline.

```bash
mlx_lm.generate \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --prompt "$PROMPT" \
  --max-tokens 128 \
  --temp 0
```

### Fused 7B Target

Purpose: measure fine-tuned target quality and speed.

```bash
mlx_lm.generate \
  --model fused-7b-lean \
  --prompt "$PROMPT" \
  --max-tokens 128 \
  --temp 0
```

### Fused 0.5B Draft Alone

Purpose: understand draft quality and failure modes. This is not expected to
match target quality.

```bash
mlx_lm.generate \
  --model fused-0.5b-lean \
  --prompt "$PROMPT" \
  --max-tokens 128 \
  --temp 0
```

### Speculative 7B Target With 0.5B Draft

Purpose: measure speed of target-equivalent generation under speculative
decoding.

```bash
mlx_lm.generate \
  --model fused-7b-lean \
  --draft-model fused-0.5b-lean \
  --num-draft-tokens 5 \
  --prompt "$PROMPT" \
  --max-tokens 128 \
  --temp 0
```

## Optional Frontier Baseline

Only include if API budget allows.

Runs:

- no examples
- few-shot examples
- many-shot examples

Report:

- compile pass rate
- input tokens
- output tokens
- estimated cost
- latency

Do not compare a cloud model with extra examples against a local model without
noting the extra context and cost.

## Correctness Metrics

Primary metric:

- `lean_compile_pass_rate`

Secondary metrics:

- exact tactic match rate
- forbidden-token rejection rate
- timeout rate
- parse/extraction failure rate
- compiler error category counts

Forbidden tokens:

- `sorry`
- `admit`
- `by sorry`
- placeholder markers such as `<TODO>`

Rules:

- A tactic with forbidden tokens is a failure even if Lean would accept it.
- A model output that cannot be extracted as a tactic is a failure.
- A timeout is a failure and should be counted separately.

## Speed Metrics

Capture:

- prompt tokens
- generated tokens
- elapsed seconds
- prompt tokens/sec if available
- generation tokens/sec
- peak memory if available

For speculative decoding also capture, if MLX exposes it:

- draft tokens proposed
- draft tokens accepted
- acceptance rate

If MLX does not expose acceptance rate, do not invent it. Report only observable
tokens/sec and latency.

## Required Output Files

Each run writes:

```text
reports/lean_eval/<run_name>/
  predictions.jsonl
  summary.json
  compiler_errors.jsonl
  run_config.json
```

Each prediction row should include:

```json
{
  "index": 0,
  "prompt": "...",
  "expected_tactic": "...",
  "generated_text": "...",
  "generated_tactic": "...",
  "forbidden_token": false,
  "lean_pass": true,
  "lean_stdout": "",
  "lean_stderr": "",
  "elapsed_seconds": 0.0,
  "generated_tokens": 0,
  "tokens_per_second": 0.0
}
```

The top-level report writes:

```text
reports/lean_eval/REPORT.md
```

## Report Requirements

The report must include:

- date and machine
- git commit or dirty-worktree note
- model names and revisions
- dataset name and split sizes
- training settings summary
- full benchmark table
- representative successes
- representative failures
- limitations

Benchmark table columns:

```text
run | records | compile_pass_rate | exact_match_rate | mean_latency_s | p50_latency_s | p95_latency_s | tokens_per_s | notes
```

## Pass Gates

A full eval is considered valid only if:

- all compared runs use the same test split
- all runs use the same prompt template
- all runs use deterministic decoding unless explicitly labeled otherwise
- Lean verification runs automatically
- generated artifacts are outside git-tracked source
- report contains no absolute `/home/<user>/...` paths

## Claims Allowed

Allowed if supported:

- "Fine-tuning improves Lean compiler pass rate from A to B."
- "Speculative decoding changes tokens/sec from X to Y on this machine."
- "The draft model alone is weaker/stronger by compiler pass rate."

Not allowed without additional evidence:

- "The speculative model is more correct than the target."
- "The output is mathematically perfect" unless Lean accepted it in the exact
  reconstructed context and forbidden tokens were absent.
- "This speedup generalizes to all Macs."
- "The cloud model cannot solve this" without a fair prompt/cost baseline.

