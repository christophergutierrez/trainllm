# Lean Speculative Decoding Docs

Start here if you are a clean agent.

Read in this order:

1. `AGENT_START.md` - kickoff brief and first task.
2. `PLAN.md` - human-readable phases, milestones, and pass gates.
3. `PLAN.yaml` - machine-readable milestone ledger.
4. `SUBAGENT_LOOP.md` - how to parallelize Sonnet/Haiku/gpt-5.5 work.
5. `FULL_EVAL.md` - final correctness and speed evaluation protocol.

The first implementation target is Phase 0, Milestone 0.1:

```bash
uv run train.py --help
uv run train.py --dry-run
```

Do not begin real training until Phase 0 and Phase 1 pass.

