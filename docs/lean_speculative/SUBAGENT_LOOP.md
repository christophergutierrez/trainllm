# Subagent Loop Design

This document describes how to run the plan with parallel subagents without
wasting high-tier model calls.

## Roles

### Coordinator

The main agent owns:

- milestone selection
- final decisions
- file edits that integrate multiple subtasks
- deciding when a phase is ready for gpt-5.5 review

### Haiku / Cheap Agents

Use for bounded, read-only or low-risk tasks:

- inspect command output and summarize the failure
- check docs for stale paths
- inspect generated bundle contents
- verify that all tests listed in `PLAN.yaml` have results
- cold-read instructions and produce a run plan
- compare two JSON reports for schema drift

Do not use cheap agents for:

- changing training code
- changing Lean verification semantics
- debugging GPU memory issues without a clear log

### Sonnet / Mid-Tier Agents

Use for implementation:

- adding CLI to `train.py`
- writing `prepare_lean_data.py`
- writing Lean verification scripts
- writing benchmark runners
- debugging failed tests
- implementing packaging scripts

### gpt-5.5 Reviewer

Use after a phase, or when stuck:

- architecture review
- correctness review
- artifact hygiene review
- repeated failure diagnosis
- final report review

## Loop Structure

For each milestone:

1. Coordinator assigns implementation to a Sonnet agent if code changes are
   needed.
2. In parallel, a Haiku agent can inspect docs/tests for the same milestone.
3. Sonnet returns changed files and test output.
4. Coordinator runs or verifies the required tests.
5. If tests pass, a Haiku agent checks for artifact hygiene.
6. If the milestone completes a phase, send phase diff and test log to gpt-5.5.
7. Apply review fixes.
8. Move to the next milestone.

## Failure Loop

When a test fails:

1. Save the failing command and exact output.
2. Ask a Haiku agent to summarize the failure if the log is long.
3. Ask a Sonnet agent to fix the smallest likely cause.
4. Re-run only the failing test first.
5. Re-run the full milestone test list after the failing test passes.
6. If the same failure repeats 3 times, ask gpt-5.5 for diagnosis.
7. Stop after 10 loops on the same blocker.

## Parallelization Opportunities

Safe to run in parallel:

- Phase 0.2 MLX audit and Phase 0.3 git hygiene audit.
- Data schema tests and documentation updates.
- Lean safety-check tests and report-format tests.
- Bundle file-list validation and cold-doc review.
- Report review and generated artifact hygiene checks.

Do not run in parallel:

- two agents editing `train.py`
- two agents editing the same runner script
- training jobs competing for the same GPU
- fusion jobs writing to the same output directory

## Status Record

Each milestone should produce a short status record:

```json
{
  "milestone": "1.1",
  "status": "pass",
  "changed_files": ["prepare_lean_data.py", "tests/test_prepare_lean_data.py"],
  "tests": [
    {"command": "uv run pytest tests/test_prepare_lean_data.py", "status": "pass"}
  ],
  "artifacts": ["data/lean_stat/train.jsonl"],
  "notes": "Generated data is ignored by git."
}
```

Store status records under an ignored run directory such as:

```text
runs/lean_speculative/<timestamp>/
```

## gpt-5.5 Phase Review Packet

Send:

- phase objective
- `git diff --stat`
- full diff for source files
- milestone status records
- failing or skipped tests
- known assumptions

Use this review ask:

```text
Review for correctness, reproducibility, artifact hygiene, and unsupported
claims. Findings first, ordered by severity. Include exact file/line references.
```

