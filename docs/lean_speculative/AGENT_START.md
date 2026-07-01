# Clean Agent Start Brief

You are starting work on the Lean speculative decoding example in `trainllm`.

Read these files first:

1. `docs/lean_speculative/PLAN.md`
2. `docs/lean_speculative/PLAN.yaml`
3. `docs/lean_speculative/FULL_EVAL.md`
4. `peft_to_mlx.py`
5. `train.py`
6. `tests/test_peft_to_mlx.py`

## Objective

Build an end-to-end workflow where training happens on the GB10 machine and the
final fused MLX models can run on either the GB10 or a Mac laptop.

The preferred next target workflow trains:

- target model adapter: `Qwen/Qwen2.5-Coder-14B-Instruct`
- draft model adapter: `Qwen/Qwen2.5-Coder-0.5B-Instruct`

Keep the 7B adapter path as a fallback baseline and comparison point.

Then it converts PEFT adapters to MLX adapters, fuses them, and benchmarks:

- base 7B
- fused 7B
- fused 0.5B
- fused 7B with fused 0.5B as speculative draft model

Correctness must be verified with Lean, not manual inspection.

## Important Existing Context

- `train.py` currently uses config/env-driven Unsloth training.
- It prevents plateau early stop before roughly one epoch. Preserve that
  behavior unless you have a better implementation.
- `peft_to_mlx.py` exists because MLX does not consume raw PEFT adapters.
- `mlx_lm.fuse` should receive MLX-format adapters, not raw PEFT adapters.
- Keep massive artifacts out of git.

## First Task

Start with Phase 0 from `PLAN.md`.

Recommended first implementation milestone:

1. Add a CLI to `train.py` while preserving current config defaults.
2. Add `--dry-run`.
3. Ensure these commands work:

```bash
uv run train.py --help
uv run train.py --dry-run
uv run train.py --dry-run \
  --base-model Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --output-dir /tmp/trainllm-dry-run \
  --train-data /tmp/fake.jsonl \
  --max-steps 20
```

Do not start real training until Phase 0 and Phase 1 tests pass.

## Agent Tiering

Use cheap Haiku-style agents for:

- reading docs and summarizing gaps
- checking generated file lists
- checking command outputs
- reviewing reports for absolute paths or stale instructions
- cold-run documentation validation

Use Sonnet-style agents for:

- code implementation
- debugging failed tests
- writing Lean verification harnesses
- training/conversion/fusion orchestration

Use gpt-5.5 review for:

- architecture review after each phase
- code review before merging phase work
- diagnosing repeated failures
- validating correctness claims in reports

## Loop Protocol

For each milestone:

1. Read the milestone in `PLAN.md`.
2. Implement only what that milestone needs.
3. Run every test listed for the milestone.
4. Save command output or a concise log summary.
5. If tests fail, run a debug loop.
6. If the same blocker survives 3 loops, ask gpt-5.5 for diagnosis.
7. Stop after 10 loops on the same blocker.
8. Do not move to the next milestone until tests pass.

## Review Prompt For gpt-5.5

Use this prompt after a phase is implemented:

```text
Review this phase against docs/lean_speculative/PLAN.md and PLAN.yaml.
Prioritize correctness bugs, broken assumptions, missing tests, artifact hygiene,
and claims that are stronger than the evidence. Do not focus on style unless it
affects maintainability or reproducibility.
```

## Cold-Agent Prompt

Use this prompt for documentation validation:

```text
You have only this repository and the files in docs/lean_speculative.
Read the plan and tell me the exact commands you would run to complete the next
milestone. Then list blockers, missing commands, or likely failures. Do not
modify files.
```

## Definition Of Done

The project is done when:

- data preparation is deterministic and tested
- both adapters train on GB10
- both adapters convert to MLX format
- both fused models run
- target-only and speculative decoding are benchmarked
- Lean compiler verification is automated
- a report gives correctness and speed results
- a clean bundle can be followed by a cold agent without internal blockers
