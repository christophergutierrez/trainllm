# Known Issues and Limitations

## Active Issues

### No per-request timeout in eval.py
**Severity:** Medium | **Component:** eval.py

If vLLM hangs on a single inference request (long input, GPU memory pressure), the eval process stalls on that record indefinitely. The WatchdogProcess wall_timeout eventually kills the entire eval, but partial results are lost.

**Workaround:** The wall_timeout (default 3600s) acts as a coarse safety net.

**Fix:** Add `timeout=60` to the OpenAI client call, catch `APITimeoutError`, log the record as ERROR, continue to next record.

---

### WS broadcast can block on slow clients
**Severity:** Low | **Component:** web/backend/ws.py

`await ws.send_text()` inside the async lock means a slow (but not dead) WebSocket client blocks all broadcasts to other clients. The stale-client cleanup catches dead connections but not slow ones.

**Impact:** Only matters with multiple remote dashboard clients. Single local browser is fine.

**Fix:** Add a per-client send timeout or switch to a queue-based broadcast pattern.

---

### Events file read on every diagnostics request
**Severity:** Low | **Component:** web/backend/routes/diagnostics.py

`_read_events()` reads the entire JSONL file on every `/api/diagnostics/pipeline-health`, `/api/diagnostics/timing`, etc. call. At current scale (~2000 events per run) this is imperceptible, but would degrade with very long training runs (10k+ steps).

**Fix:** Cache parsed events in memory (the backend already tails the file for WS broadcast — reuse that data).

---

### Agent presence TTL may be too aggressive
**Severity:** Low | **Component:** web/backend/agent_bridge.py

The presence file TTL is 5 minutes. If a Claude Code session is busy thinking (e.g., large code review, complex reasoning) for more than 5 minutes without refreshing the presence file, the dashboard incorrectly shows "AI Disabled."

**Workaround:** The `agent_relay.py --presence` daemon refreshes every 60s independently of the session's activity.

**Fix:** Increase TTL to 10 minutes, or have the daemon refresh more frequently.

---

### No WebSocket exponential backoff
**Severity:** Low | **Component:** web/frontend/src/lib/ws.ts

Fixed 3-second reconnect interval. Under sustained backend failure, this creates steady reconnect churn. Not harmful for local use but wasteful.

**Fix:** Implement exponential backoff (3s → 6s → 12s → max 30s), reset on successful connection.

---

## Architectural Debt

### cycle.py is too large (1656 lines)
The orchestrator handles backup, vLLM lifecycle, training, merge, eval, checkpoint selection, and reporting. This makes it harder to test individual steps in isolation and increases the blast radius of changes.

**Planned refactor:** See the cycle.py refactoring plan (when ready to restructure without active training running).

---

### No linting or formatting enforcement
No ruff, mypy, or prettier configured. Code quality is maintained manually. Fine for solo development but would need enforcement with multiple contributors.

**Plan:** Add `ruff check .` to a pre-commit hook or Makefile target. Skip mypy (ML codebases fight it constantly with torch/unsloth types).

---

### Test coverage is selective
Tests exist for: config, bundle, eval_utils, data prep, event emitter, WSD scheduler.

Tests do NOT exist for: cycle.py orchestration, web routes, WebSocket handlers, agent relay, merge.py.

---

### No authentication on web dashboard
All endpoints are open on the network. Fine for local development on a private machine. Would need auth for any remote/shared access.

---

## Resolved Issues

- **WebSocket reconnect duplicated chart data** — Fixed: stores.ts now deduplicates by step number on replay.
- **`die()` not emitting error events** — Fixed: `emit_error()` called in `die()` so all fatal exits are observable on the dashboard.
- **nvidia-smi returns [N/A] for GPU memory on GB10** — Fixed: diagnostics endpoint falls back to `free -m` for unified memory systems.
- **Stale training data shown after new run starts** — Fixed: `step_start` event resets all stores.
