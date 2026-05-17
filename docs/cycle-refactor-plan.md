# cycle.py Refactoring Plan

## Current State

`cycle.py` is 1656 lines with 40 functions/classes spanning 6 distinct concerns. The largest functions are:

| Function | Lines | Concern |
|----------|-------|---------|
| `step_report()` | 206 | Reporting |
| `main()` | 204 | Orchestration |
| `step_train()` | 148 | Training |
| `WatchdogProcess` | 107 | Subprocess mgmt |
| `_launch_and_poll_vllm()` | 75 | vLLM lifecycle |
| `step_select_best_checkpoint()` | 74 | Eval/selection |
| `step_start_vllm()` | 70 | vLLM lifecycle |
| `_validate_training_data()` | 68 | Data validation |
| `_emit_result_json()` | 67 | Artifact output |
| `_wait_for_free_gpu_memory()` | 56 | GPU memory |

## Proposed Module Split

### 1. `_log.py` — Logging and utilities (42 lines)
```
_setup_logging()
log()
log_section()
die()
_elapsed()
_notify_agent()
```

**Rationale:** Every other module needs `log()` and `die()`. Extracting these first enables all other modules to import from a single place. The `emit_error` and `_notify_agent` integration stays here since `die()` calls both.

### 2. `_watchdog.py` — Subprocess management (107 lines)
```
class WatchdogProcess
```

**Rationale:** Self-contained class with no dependencies beyond `_log`. Used by training, eval, and merge steps. Already has a clean interface (cmd, label, timeouts, callback → returncode).

### 3. `_vllm.py` — vLLM lifecycle management (~340 lines)
```
_find_vllm_pids()
_find_all_vllm_pids()
_kill_process_groups()
step_stop_vllm()
_drop_page_cache()
_reload_nvidia_uvm()
_query_cuda_free_gib()
_reclaim_gpu_memory()
_wait_for_free_gpu_memory()
_launch_and_poll_vllm()
step_start_vllm()
step_start_vllm_merged()
stop_managed_vllm()
_checkpoint_dirs()
_checkpoint_module_name()
```

**Rationale:** Largest cohesive block. All related to starting/stopping/managing vLLM processes and GPU memory. No dependencies on training or eval logic. The GPU memory reclaim functions are only used before vLLM startup.

### 4. `_train_step.py` — Training orchestration (~240 lines)
```
_auto_max_steps()
_clear_stale_checkpoints()
_manifest_max_steps()
_validate_training_data()
step_train()
```

**Rationale:** Everything specific to running the training subprocess, including data validation, step calculation, loss parsing, and OOM retry. Uses `WatchdogProcess` from `_watchdog.py`.

### 5. `_eval_step.py` — Evaluation and checkpoint selection (~140 lines)
```
find_latest_eval()
find_prev_eval()
step_eval()
step_select_best_checkpoint()
step_emit_synth_status()
step_llm_judge()
```

**Rationale:** All eval-related orchestration. Uses `WatchdogProcess`. The best-checkpoint logic lives here because it's fundamentally an eval concern (run N evals, pick the winner).

### 6. `_report.py` — Report generation (~220 lines)
```
_load_eval()
_score_bar()
step_report()
```

**Rationale:** Pure output formatting. No side effects beyond logging. Largest single function (206 lines). Easy to extract since it only reads eval JSON files and prints formatted output.

### 7. `_artifacts.py` — Result file emission (~80 lines)
```
_emit_result_json()
step_backup()
```

**Rationale:** Both deal with writing structured output files (result.json, adapter backups). Small but logically distinct from orchestration.

### 8. `cycle.py` — Orchestrator (reduced to ~280 lines)
```
parse_args()
main()
step_merge()  # small enough to stay, or move to its own file later
```

**Rationale:** `main()` becomes a pure orchestrator that imports steps from the other modules and sequences them. The merge step (31 lines) can stay or be extracted later.

## Dependency Graph

```
_log.py (no deps)
   ↑
_watchdog.py (imports _log)
   ↑
_vllm.py (imports _log, _watchdog, _config)
_train_step.py (imports _log, _watchdog, _config, _callbacks)
_eval_step.py (imports _log, _watchdog, _config)
_report.py (imports _log)
_artifacts.py (imports _log, _config)
   ↑
cycle.py (imports all of the above)
```

No circular dependencies. Each module only depends on `_log` + `_config` + optionally `_watchdog`.

## Migration Strategy

1. **Extract `_log.py` first** — enables all other extractions.
2. **Extract `_watchdog.py`** — no behavior change, class is self-contained.
3. **Extract `_vllm.py`** — largest win, removes 340 lines from cycle.py.
4. **Extract `_report.py`** — pure output, zero risk.
5. **Extract `_train_step.py`** and `_eval_step.py`** — these share the OOM retry and checkpoint logic, so do them together.
6. **Extract `_artifacts.py`** — small, clean boundary.

Each extraction can be done as an independent commit. Tests should pass after each step (existing tests don't import from cycle.py directly).

## What NOT to Change

- **Don't change the public interface.** `cycle.py` is still the entry point. `python cycle.py --canary` still works.
- **Don't restructure the pipeline flow.** The step sequence stays the same.
- **Don't introduce abstractions.** No step runner framework, no plugin system. Just file-level separation.
- **Don't rename functions.** Consumers (like the web backend's cycle route) may reference step names.

## When to Do This

- After training completes (cycle.py is the running process)
- When next adding a major feature to the pipeline
- Not urgent — the file is large but not unmanageable for a solo developer
