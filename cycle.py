#!/usr/bin/env python3
"""
Full training cycle: backup → train → serve → eval (fine-tuned + base) → report.

Usage:
    python ~/trainLLM/cycle.py                             # use config.yaml defaults
    python ~/trainLLM/cycle.py --version 20260416         # training_20260416.jsonl + holdout_20260416.jsonl
    python ~/trainLLM/cycle.py --skip-train               # serve + eval only
    python ~/trainLLM/cycle.py --skip-train --skip-serve  # eval only
    python ~/trainLLM/cycle.py --holdout ~/data/holdout_v2.jsonl  # override holdout
    python ~/trainLLM/cycle.py --auto-steps                       # compute steps from data size
    python ~/trainLLM/cycle.py --no-best-checkpoint               # skip checkpoint selection

Reading the report:
    - Fine-tuned avg > base avg by ≥0.05 means training is helping.
    - If fine-tuned avg < 0.35 with positive delta, check max_steps first.
    - Convention breakdown (worst-first) shows which patterns need more training data.

Logs: <base_dir>/logs/cycle_<timestamp>.log
"""

import argparse
import atexit
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "handoff"))
import _config
import emit_synth_status
cfg = _config.load()

# ── Paths (all derived from config) ───────────────────────────────────────────

LORA_DIR     = cfg.lora_dir
FINAL_DIR    = cfg.final_dir
EVALS_DIR    = cfg.evals_dir
LOGS_DIR     = cfg.logs_dir
DATA_DIR     = cfg.data_dir
TRAIN_SCRIPT = Path(__file__).parent / "train.py"
EVAL_SCRIPT  = Path(__file__).parent / "eval.py"
VLLM_URL     = cfg.vllm_url
BASE_MODEL   = cfg.model
LORA_MODEL   = cfg.adapter_name
HF_HOME      = str(cfg.hf_home)
UNSLOTH_PYTHON = str(cfg.unsloth_python)

# ── Data files ─────────────────────────────────────────────────────────────────

TRAIN_DATA: Path = cfg.train_data
HOLDOUT:    Path = cfg.holdout

# ── Timeouts ──────────────────────────────────────────────────────────────────

TRAIN_SILENCE_TIMEOUT = cfg.train_silence_timeout
VLLM_STARTUP_TIMEOUT  = cfg.vllm_startup_timeout
VLLM_POLL_INTERVAL    = cfg.vllm_poll_interval
EVAL_TIMEOUT          = cfg.eval_timeout

# ── Training loss regex ───────────────────────────────────────────────────────

_LOSS_RE = re.compile(r"'loss':\s*'?([0-9.eE+-]+)'?.*?'epoch':\s*'?([0-9.eE+-]+)'?")

# ── Logging ───────────────────────────────────────────────────────────────────

TIMESTAMP: str      = ""
LOG_PATH: Path|None = None

_log_file = None
_log_lock = threading.Lock()


def _setup_logging() -> None:
    global TIMESTAMP, LOG_PATH, _log_file
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    LOG_PATH  = LOGS_DIR / f"cycle_{TIMESTAMP}.log"
    _log_file = open(LOG_PATH, "w", buffering=1)
    atexit.register(_log_file.close)


def log(msg: str, level: str = "INFO") -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [{level}] {msg}"
    with _log_lock:
        print(line, flush=True)
        if _log_file is not None:
            _log_file.write(line + "\n")


def log_section(title: str) -> None:
    bar = "─" * 60
    log(f"\n{bar}")
    log(f"  {title}")
    log(f"{bar}")


def die(msg: str) -> None:
    log(msg, "FATAL")
    if LOG_PATH:
        log(f"Full log: {LOG_PATH}", "FATAL")
    sys.exit(1)


def _elapsed(t0: float) -> str:
    s = int(time.time() - t0)
    m, s = divmod(s, 60)
    return f"{m}m {s}s" if m else f"{s}s"


# ── Subprocess helpers ────────────────────────────────────────────────────────

class WatchdogProcess:
    """
    Run a subprocess, stream its stdout, and enforce timeouts.

    silence_timeout: kills if no stdout line arrives within this many seconds.
    wall_timeout:    kills after this many total seconds regardless of output.
    line_callback:   called for every stdout line; use for parsing loss values, etc.
    """

    def __init__(self, cmd: list[str], label: str,
                 silence_timeout: int = 0, wall_timeout: int = 0,
                 env: dict | None = None,
                 line_callback: Callable[[str], None] | None = None):
        self.cmd             = cmd
        self.label           = label
        self.silence_timeout = silence_timeout
        self.wall_timeout    = wall_timeout
        self.env             = env
        self.line_callback   = line_callback
        self.proc: subprocess.Popen | None = None
        self._last_output    = time.time()
        self._hung           = False
        self._done           = threading.Event()
        self.returncode: int | None = None
        self.pid: int | None = None

    def _watchdog(self) -> None:
        start = time.time()
        while not self._done.is_set():
            time.sleep(5)
            now = time.time()
            if self.wall_timeout and (now - start) > self.wall_timeout:
                log(f"[{self.label}] WALL TIMEOUT ({self.wall_timeout}s) — killing", "WARN")
                self._hung = True
                if self.proc:
                    self.proc.kill()
                return
            if self.silence_timeout and (now - self._last_output) > self.silence_timeout:
                log(f"[{self.label}] SILENCE TIMEOUT ({self.silence_timeout}s) — likely hung — killing", "WARN")
                self._hung = True
                if self.proc:
                    self.proc.kill()
                return

    def run(self) -> int:
        full_env = {**os.environ, **(self.env or {})}
        log(f"$ {shlex.join(self.cmd)}")
        self.proc = subprocess.Popen(
            self.cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=full_env,
        )
        self.pid = self.proc.pid
        log(f"[{self.label}] PID {self.pid}")

        pid_file = LOGS_DIR / f"{self.label.replace(' ', '_')}_{TIMESTAMP}.pid"
        pid_file.write_text(str(self.pid))

        watcher = threading.Thread(target=self._watchdog, daemon=True)
        watcher.start()

        try:
            for line in self.proc.stdout:
                self._last_output = time.time()
                stripped = line.rstrip()
                if self.line_callback:
                    self.line_callback(stripped)
                log(f"  {stripped}", self.label)
        except Exception as exc:
            log(f"[{self.label}] stdout read error: {exc}", "WARN")

        self.proc.wait()
        self._done.set()
        self.returncode = self.proc.returncode
        pid_file.unlink(missing_ok=True)
        return self.returncode

    @property
    def hung(self) -> bool:
        return self._hung


# ── Training data validation ──────────────────────────────────────────────────

def _validate_training_data() -> int:
    """
    Validate TRAIN_DATA is in ShareGPT format and return record count.

    Unsloth requires ShareGPT format:
        {"conversations": [{"from": "human", "value": "..."},
                           {"from": "gpt",   "value": "..."}]}

    Wrong format silently produces a bad model. This catches the most common
    mistakes before wasting GPU hours.
    """
    if not TRAIN_DATA.exists():
        die(f"Training data not found: {TRAIN_DATA}\n  Copy your ShareGPT JSONL to {DATA_DIR}/")

    records = 0
    errors: list[str] = []
    with TRAIN_DATA.open() as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            records += 1
            if records > 3:  # Intentional: only validate first 3 records for speed, not comprehensive schema check
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"  Line {i + 1}: JSON parse error: {e}")
                continue
            if "conversations" not in rec:
                keys = list(rec.keys())[:6]
                errors.append(
                    f"  Record {records}: missing 'conversations' key (got: {keys}). "
                    "Expected ShareGPT format."
                )
            else:
                convs = rec["conversations"]
                if not isinstance(convs, list) or not convs:
                    errors.append(f"  Record {records}: 'conversations' must be a non-empty list")
                elif not isinstance(convs[0], dict) or "from" not in convs[0] or "value" not in convs[0]:
                    got = list(convs[0].keys()) if isinstance(convs[0], dict) else type(convs[0]).__name__
                    errors.append(
                        f"  Record {records}: conversations[0] missing 'from'/'value' keys (got: {got}). "
                        "Expected ShareGPT format."
                    )

    if errors:
        log("Training data format validation FAILED:", "WARN")
        for e in errors:
            log(e, "WARN")
        die(
            f"{TRAIN_DATA.name} is not in ShareGPT format.\n"
            '  Expected: {"conversations": [{"from": "human", "value": "..."},\n'
            '                               {"from": "gpt",   "value": "..."}]}'
        )

    if records < 100:
        log(f"WARNING: only {records} training records — model will likely underfit.", "WARN")
    elif records < 500:
        log(f"WARNING: only {records} records — consider adding more examples.", "WARN")
    else:
        log(f"Training data: {records} records — format OK")

    return records


# ── Step 1: Backup ────────────────────────────────────────────────────────────

def step_backup() -> None:
    log_section("STEP 1: Backup current LoRA weights")
    date_str   = datetime.now().strftime("%Y%m%d")
    backup_dir = LORA_DIR / f"final-v{date_str}"

    if not FINAL_DIR.exists():
        log(f"No existing adapter at {FINAL_DIR} — skipping backup", "WARN")
        return

    candidate = backup_dir
    suffix = 0
    while candidate.exists():
        suffix += 1
        candidate = LORA_DIR / f"final-v{date_str}-{suffix}"
    backup_dir = candidate

    log(f"Copying {FINAL_DIR} → {backup_dir}")
    shutil.copytree(FINAL_DIR, backup_dir)
    log(f"Backup complete: {backup_dir}")


# ── Step 2: Stop vLLM ────────────────────────────────────────────────────────

def step_stop_vllm() -> None:
    log_section("STEP 2: Stop any running vLLM server")
    result = subprocess.run(["pgrep", "-f", "vllm serve"], capture_output=True, text=True)
    pids = result.stdout.strip().split()
    if not pids or not any(p.strip() for p in pids):
        log("No vllm serve process found — nothing to stop")
        return
    for pid in pids:
        pid = pid.strip()
        if not pid:
            continue
        log(f"Sending SIGTERM to vllm serve PID {pid}")
        try:
            os.kill(int(pid), signal.SIGTERM)
        except ProcessLookupError:
            log(f"PID {pid} already gone")
    time.sleep(3)
    result = subprocess.run(["pgrep", "-f", "vllm serve"], capture_output=True, text=True)
    pids = result.stdout.strip().split()
    for pid in pids:
        pid = pid.strip()
        if not pid:
            continue
        log(f"Still running — sending SIGKILL to PID {pid}", "WARN")
        try:
            os.kill(int(pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
    log("vLLM stopped")


# ── Step 3: Train ─────────────────────────────────────────────────────────────

def _auto_max_steps(n_records: int, target_epochs: int = 4) -> int:
    eff_batch = cfg.training.batch_size * cfg.training.gradient_accumulation_steps
    steps_per_epoch = max(1, n_records // eff_batch)
    return max(1, target_epochs * steps_per_epoch)


def _clear_stale_checkpoints() -> None:
    stale = sorted(LORA_DIR.glob("checkpoint-*"))
    if not stale:
        return
    log(f"Removing {len(stale)} stale checkpoint dir(s) from prior runs")
    for d in stale:
        shutil.rmtree(d, ignore_errors=True)


def _manifest_max_steps() -> int | None:
    """Return meta.max_steps from data/manifest.yaml if set, else None."""
    path = DATA_DIR / "manifest.yaml"
    if not path.exists():
        return None
    try:
        import yaml
        m = yaml.safe_load(path.read_text()) or {}
    except Exception as exc:
        log(f"Could not read {path}: {exc}", "WARN")
        return None
    v = ((m.get("meta") or {}).get("max_steps"))
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        log(f"Ignoring non-integer meta.max_steps={v!r} in {path}", "WARN")
        return None


def step_train(max_steps: int | None = None, auto_steps: bool = False) -> None:
    log_section("STEP 3: Train (QLoRA via Unsloth)")
    log(f"Training script: {TRAIN_SCRIPT}")
    log(f"Training data:   {TRAIN_DATA}")
    log(f"Python:          {UNSLOTH_PYTHON}")
    log(f"Silence timeout: {TRAIN_SILENCE_TIMEOUT}s")

    if not Path(UNSLOTH_PYTHON).exists():
        die(f"Unsloth Python not found at {UNSLOTH_PYTHON}. Check paths.unsloth_python in config.yaml.")

    n_records = _validate_training_data()

    # Precedence when no CLI override is passed (--steps, --canary handled by caller):
    #   1. manifest meta.max_steps (reposynth's per-cycle recommendation)
    #   2. --auto-steps formula (4 epochs × records / effective_batch)
    #   3. config default (falls through as max_steps stays None)
    if max_steps is None:
        manifest_steps = _manifest_max_steps()
        if manifest_steps is not None:
            max_steps = manifest_steps
            eff_batch = cfg.training.batch_size * cfg.training.gradient_accumulation_steps
            epochs = max_steps / max(1, n_records // eff_batch)
            log(f"max_steps from manifest: {max_steps} steps (~{epochs:.1f} epochs on "
                f"{n_records} records)")
        elif auto_steps:
            max_steps = _auto_max_steps(n_records)
            eff_batch = cfg.training.batch_size * cfg.training.gradient_accumulation_steps
            epochs = max_steps / max(1, n_records // eff_batch)
            log(f"Auto max_steps: {n_records} records × 4 epochs / {eff_batch} batch "
                f"→ {max_steps} steps (~{epochs:.1f} epochs)")

    _clear_stale_checkpoints()

    loss_points: list[tuple[float, float]] = []
    warned_loss = False

    def _loss_callback(line: str) -> None:
        nonlocal warned_loss
        m = _LOSS_RE.search(line)
        if not m:
            return
        try:
            loss  = float(m.group(1))
            epoch = float(m.group(2))
        except ValueError:
            return
        loss_points.append((epoch, loss))
        if not warned_loss and epoch > 0.2 and loss > 1.5:
            log(
                f"  EARLY WARNING: loss={loss:.3f} at epoch={epoch:.2f}. "
                "Expected < 1.5 by epoch 0.2. Check: data format, chat_template in config.yaml, LR.",
                "WARN",
            )
            warned_loss = True

    env = {
        "PYTHONUNBUFFERED": "1",
        "TRAIN_DATA": str(TRAIN_DATA),
        "OUTPUT_DIR": str(LORA_DIR),
    }
    if max_steps is not None:
        env["MAX_STEPS"] = str(max_steps)
        log(f"MAX_STEPS override: {max_steps}")

    wp = WatchdogProcess(
        cmd=[UNSLOTH_PYTHON, "-u", str(TRAIN_SCRIPT)],
        label="TRAIN",
        silence_timeout=TRAIN_SILENCE_TIMEOUT,
        env=env,
        line_callback=_loss_callback,
    )
    rc = wp.run()

    _CANARY_TARGET = 0.6  # "good" band threshold; used for step estimation below

    if loss_points:
        first_loss  = loss_points[0][1]
        final_loss  = loss_points[-1][1]
        final_epoch = loss_points[-1][0]
        log(f"Training loss: {first_loss:.3f} → {final_loss:.3f} "
            f"(over {final_epoch:.1f} epochs, {len(loss_points)} steps logged)")
        if final_loss > 1.0:
            log("  WARNING: Final loss > 1.0 — model has not converged. Consider increasing max_steps.", "WARN")
        elif final_loss < 0.15:
            log("  WARNING: Final loss < 0.15 — may be overfit. Verify holdout scores.", "WARN")

        if max_steps is not None and len(loss_points) >= 3:
            loss_drop = first_loss - final_loss
            if loss_drop > 0 and final_loss > _CANARY_TARGET:
                rate = loss_drop / max_steps          # loss units per step
                extra = int((final_loss - _CANARY_TARGET) / rate)
                suggested = int((max_steps + extra) * 1.2 / 100 + 1) * 100  # round up to nearest 100
                log(
                    f"  Step estimate: {max_steps} steps → loss {first_loss:.3f}→{final_loss:.3f} "
                    f"({rate:.5f}/step). To reach {_CANARY_TARGET}: ~{max_steps + extra} more steps. "
                    f"Suggested: --steps {suggested}",
                    "INFO",
                )
            elif loss_drop > 0 and final_loss <= _CANARY_TARGET:
                log(f"  Loss already at {final_loss:.3f} (≤{_CANARY_TARGET}) — {max_steps} steps may be sufficient.")
            else:
                log("  Loss did not decrease — check data format and chat_template before a full run.", "WARN")
    else:
        log("Could not parse training loss — check log for trainer output", "WARN")

    if wp.hung:
        die(
            f"Training hung — no output for {TRAIN_SILENCE_TIMEOUT}s.\n"
            "  Check GPU: nvidia-smi\n"
            "  Check dmesg: sudo dmesg | tail -20\n"
            f"  Full log: {LOG_PATH}"
        )

    if rc != 0:
        die(
            f"Training exited with code {rc}.\n"
            f"  Data file: {TRAIN_DATA.name}\n"
            "  Is it in ShareGPT format? (conversations → from/value)\n"
            f"  See log above for details."
        )

    if not FINAL_DIR.exists():
        die(f"Training completed but {FINAL_DIR} not found — adapter was not saved.")

    safetensors = list(FINAL_DIR.glob("*.safetensors"))
    if not safetensors:
        log(f"WARNING: no .safetensors files in {FINAL_DIR} — adapter may be corrupt", "WARN")
    else:
        total_mb = sum(f.stat().st_size for f in safetensors) / 1_048_576
        log(f"Adapter saved: {len(safetensors)} shard(s), {total_mb:.1f} MB — {FINAL_DIR}")

    log("Training complete.")


# ── Step 4: Start vLLM ────────────────────────────────────────────────────────

def _checkpoint_dirs() -> list[Path]:
    def step_num(p: Path) -> int:
        try:
            return int(p.name.split("-")[1])
        except (IndexError, ValueError):
            return -1
    return sorted(
        (p for p in LORA_DIR.glob("checkpoint-*") if p.is_dir() and step_num(p) > 0),
        key=step_num,
    )


def _checkpoint_module_name(cp: Path) -> str:
    return f"{LORA_MODEL}-ckpt{cp.name.split('-')[1]}"


def step_start_vllm(include_checkpoints: bool = False) -> subprocess.Popen:
    log_section("STEP 4: Start vLLM inference server")

    lora_modules = [f"{LORA_MODEL}={FINAL_DIR}"]
    if include_checkpoints:
        for cp in _checkpoint_dirs():
            lora_modules.append(f"{_checkpoint_module_name(cp)}={cp}")
        if len(lora_modules) > 1:
            log(f"Loading {len(lora_modules)} LoRA modules (final + "
                f"{len(lora_modules)-1} checkpoint(s)) for best-checkpoint selection")

    cmd = [
        "vllm", "serve", BASE_MODEL,
        "--dtype", "bfloat16",
        "--gpu-memory-utilization", str(cfg.vllm_gpu_memory_util),
        "--enforce-eager",   # required on Blackwell (GB10): avoids torch.compile hang
        "--enable-lora",
        "--lora-modules", *lora_modules,
        "--max-lora-rank", str(cfg.training.lora_rank),
        "--port", str(cfg.vllm_port),
    ]
    env = {**os.environ, "HF_HOME": HF_HOME}

    log(f"$ {shlex.join(cmd)}")
    log(f"Startup timeout: {VLLM_STARTUP_TIMEOUT}s  (poll every {VLLM_POLL_INTERVAL}s)")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    log(f"vLLM PID: {proc.pid}")
    pid_file = LOGS_DIR / f"vllm_{TIMESTAMP}.pid"
    pid_file.write_text(str(proc.pid))

    failed_event = threading.Event()

    def stream_logs():
        for line in proc.stdout:
            log(f"  {line.rstrip()}", "VLLM")
            if proc.poll() is not None:
                failed_event.set()
                break

    threading.Thread(target=stream_logs, daemon=True).start()

    start    = time.time()
    last_poll = 0.0

    while True:
        elapsed = time.time() - start

        if elapsed > VLLM_STARTUP_TIMEOUT:
            log(f"vLLM did not become ready within {VLLM_STARTUP_TIMEOUT}s — killing", "FATAL")
            log("  Check: --enforce-eager set? nvidia-smi? HF model cache present?", "FATAL")
            proc.kill()
            pid_file.unlink(missing_ok=True)
            die("vLLM startup timeout")

        if failed_event.is_set() or proc.poll() is not None:
            rc = proc.poll()
            log(f"vLLM process exited unexpectedly with code {rc}", "FATAL")
            pid_file.unlink(missing_ok=True)
            die("vLLM exited during startup — check log above")

        if time.time() - last_poll >= VLLM_POLL_INTERVAL:
            last_poll = time.time()
            log(f"  Polling {VLLM_URL}/v1/models ... (elapsed {int(elapsed)}s)")
            try:
                with urllib.request.urlopen(f"{VLLM_URL}/v1/models", timeout=5) as resp:
                    data      = json.loads(resp.read())
                    model_ids = [m["id"] for m in data.get("data", [])]
                    log(f"  Server ready! Models: {model_ids}")
                    if LORA_MODEL not in model_ids:
                        log(
                            f"  WARNING: '{LORA_MODEL}' not in model list {model_ids}. "
                            f"Check --lora-modules and that {FINAL_DIR} exists.",
                            "WARN",
                        )
                    pid_file.unlink(missing_ok=True)
                    return proc
            except Exception as exc:
                log(f"  Not ready yet: {exc}")

        time.sleep(5)


# ── Step 5: Evaluate ──────────────────────────────────────────────────────────

def find_latest_eval(model: str) -> Path | None:
    safe = model.replace("/", "_")
    matches = list(EVALS_DIR.glob(f"*_{safe}.json"))
    return max(matches, key=lambda p: p.stat().st_mtime) if matches else None


def find_prev_eval(model: str) -> Path | None:
    safe = model.replace("/", "_")
    matches = sorted(EVALS_DIR.glob(f"*_{safe}.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[1] if len(matches) >= 2 else None


def step_eval(model: str, label: str) -> Path | None:
    log_section(f"STEP 5: Evaluate — {label} ({model})")
    log(f"Holdout: {HOLDOUT}")
    if not HOLDOUT.exists():
        log(f"Holdout file not found: {HOLDOUT} — skipping eval", "WARN")
        return None
    env = {
        "MODEL":             model,
        "HOLDOUT":           str(HOLDOUT),
        "VLLM_URL":          VLLM_URL,
        "PYTHONUNBUFFERED":  "1",
    }
    wp = WatchdogProcess(
        cmd=[sys.executable, "-u", str(EVAL_SCRIPT)],
        label=f"EVAL:{label}",
        wall_timeout=EVAL_TIMEOUT,
        env=env,
    )
    rc = wp.run()

    if wp.hung:
        log(f"Eval hung after {EVAL_TIMEOUT}s — skipping this model", "WARN")
        return None
    if rc != 0:
        log(f"Eval exited with code {rc} — results may be incomplete", "WARN")

    result = find_latest_eval(model)
    if result:
        log(f"Eval output: {result}")
    else:
        log("Could not find eval output file", "WARN")
    return result


# ── Step 5a: Best-checkpoint selection ────────────────────────────────────────

def step_select_best_checkpoint() -> Path | None:
    """Eval each saved checkpoint + final, promote the highest-scoring to final/.

    Returns the eval JSON path for the winning adapter so step_report can use it.
    Falls back to a standard eval of final/ if no intermediate checkpoints exist.
    """
    ckpts = _checkpoint_dirs()
    if not ckpts:
        log("No intermediate checkpoints saved — evaluating final/ directly")
        return step_eval(LORA_MODEL, "fine-tuned")

    log_section(f"STEP 5a: Best-checkpoint selection ({len(ckpts) + 1} candidates)")

    candidates: list[tuple[str, Path]] = [(LORA_MODEL, FINAL_DIR)]
    for cp in ckpts:
        candidates.append((_checkpoint_module_name(cp), cp))

    scored: list[tuple[str, Path, float, Path]] = []
    for model_name, ckpt_path in candidates:
        label = "final" if ckpt_path == FINAL_DIR else ckpt_path.name
        eval_path = step_eval(model_name, label)
        if not eval_path:
            log(f"  {model_name}: eval failed or returned no file", "WARN")
            continue
        data = _load_eval(eval_path)
        if not data:
            continue
        score = data["summary"]["avg_score"]
        log(f"  {label:18s} ({model_name}): avg {score:.4f}")
        scored.append((model_name, ckpt_path, score, eval_path))

    if not scored:
        log("All checkpoint evals failed — no winner to select", "WARN")
        return None

    winner_name, winner_path, winner_score, winner_eval = max(scored, key=lambda x: x[2])
    final_score = next((s for _n, p, s, _e in scored if p == FINAL_DIR), None)

    log("")
    log(f"Winner: {winner_path.name} ({winner_name}) with avg {winner_score:.4f}")
    if winner_path == FINAL_DIR:
        log("final/ was already the best — no promotion needed")
    else:
        if final_score is not None:
            log(f"  (final/ scored {final_score:.4f}; "
                f"winner beats it by {winner_score - final_score:+.4f})")
        log(f"Promoting {winner_path} → {FINAL_DIR}")
        tmp = FINAL_DIR.with_suffix(".tmp_promote")
        if tmp.exists():
            shutil.rmtree(tmp)
        shutil.copytree(winner_path, tmp)
        shutil.rmtree(FINAL_DIR)
        tmp.rename(FINAL_DIR)
        log("final/ now holds the winning checkpoint")

    return winner_eval


# ── Step 5b: Emit synth_status.yaml ───────────────────────────────────────────

def step_emit_synth_status(ft_path: Path | None) -> Path | None:
    if not ft_path:
        return None
    log_section("STEP 5b: Emit synth_status.yaml for reposynth handoff")
    try:
        out_path = emit_synth_status.emit(ft_path)
    except Exception as exc:
        log(f"synth_status emit failed: {exc}", "WARN")
        return None
    log(f"synth_status: {out_path}")
    return out_path


# ── Step 6: Report ────────────────────────────────────────────────────────────

def _load_eval(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        log(f"Could not load {path}: {exc}", "WARN")
        return None


def _score_bar(score: float, width: int = 30) -> str:
    filled = int(score * width)
    return "[" + "█" * filled + "·" * (width - filled) + f"] {score:.2f}"


def step_report(ft_path: Path | None, base_path: Path | None, status_path: Path | None = None) -> None:
    log_section("STEP 6: Results summary and recommendations")

    ft_data   = _load_eval(ft_path)   if ft_path   else None
    base_data = _load_eval(base_path) if base_path else None

    if not ft_data and not base_data:
        log("No eval data available — cannot generate report", "WARN")
        return

    def r(s: str = "") -> None:
        log(s, "REPORT")

    r()
    r("=" * 64)
    r("  CYCLE COMPLETE — EVAL SUMMARY")
    r("=" * 64)

    if ft_data and base_data:
        ft_avg   = ft_data["summary"]["avg_score"]
        base_avg = base_data["summary"]["avg_score"]
        delta    = ft_avg - base_avg
        r()
        r(f"  Fine-tuned ({LORA_MODEL}):  {_score_bar(ft_avg)}")
        r(f"  Base model ({BASE_MODEL}):  {_score_bar(base_avg)}")
        r()
        if delta > 0.05:
            r(f"  Fine-tuning is helping:  +{delta:.2f} above base")
        elif delta < -0.05:
            r(f"  WARNING: Fine-tuning is HURTING the model: {delta:.2f} vs base")
            r("  Check: data format, LR too high, too many steps")
        else:
            r(f"  Fine-tuning has minimal effect: delta={delta:+.2f}")
            r("  Consider: more training data, higher LoRA rank, different base model")
    elif ft_data:
        r()
        r(f"  Fine-tuned ({LORA_MODEL}):  {_score_bar(ft_data['summary']['avg_score'])}")
        r("  (No base model run for comparison)")
    elif base_data:
        r()
        r(f"  Base model ({BASE_MODEL}):  {_score_bar(base_data['summary']['avg_score'])}")
        r("  (No fine-tuned model run for comparison)")

    for label, data in [("Fine-tuned", ft_data), ("Base", base_data)]:
        if not data:
            continue
        bc = data["summary"]["band_counts"]
        n  = data["meta"]["n_records"]
        r()
        r(f"  {label} band breakdown (n={n}):")
        r(f"    Excellent (≥0.8): {bc.get('EXCELLENT', 0)}")
        r(f"    Good    (0.6-0.8): {bc.get('GOOD', 0)}")
        r(f"    Partial (0.4-0.6): {bc.get('PARTIAL', 0)}")
        r(f"    Poor    (<0.4):    {bc.get('POOR', 0)}")
        if bc.get("ERROR", 0):
            r(f"    Errors:           {bc['ERROR']}  ← check eval log")

    if ft_data:
        weak = ft_data["summary"].get("weak_conventions", [])
        conv = ft_data["summary"].get("convention_breakdown", [])
        if conv:
            r()
            r("  Convention scores (fine-tuned, worst first):")
            for c in conv:
                marker = " ← WEAK" if c["avg"] < 0.6 else ""
                r(f"    {c['convention']:30s}  {c['avg']:.2f}  (n={c['n']}){marker}")
        if weak:
            r()
            r("  PRIORITY TRAINING TARGETS:")
            for i, c in enumerate(weak[:5], 1):
                r(f"    {i}. `{c['convention']}` — avg {c['avg']:.2f}")

    if ft_data and ft_path:
        prev_path = find_prev_eval(LORA_MODEL)
        prev_data = _load_eval(prev_path) if prev_path else None
        if prev_data:
            prev_avg = prev_data["summary"]["avg_score"]
            curr_avg = ft_data["summary"]["avg_score"]
            delta_vs_prev = curr_avg - prev_avg
            arrow = "▲" if delta_vs_prev > 0 else ("▼" if delta_vs_prev < 0 else "─")
            r()
            r(f"  vs previous run ({prev_path.stem[:16]}):  "
              f"{prev_avg:.2f} → {curr_avg:.2f}  {arrow}{abs(delta_vs_prev):.2f}")

            prev_by_id   = {res["id"]: res["score"] for res in prev_data.get("results", [])}
            regressions  = []
            improvements = []
            for res in ft_data.get("results", []):
                prev_score = prev_by_id.get(res["id"])
                if prev_score is None:
                    continue
                d = res["score"] - prev_score
                label = res.get("label", res["id"])
                if d <= -0.05:
                    regressions.append((res["id"], label, prev_score, res["score"], d))
                elif d >= 0.05:
                    improvements.append((res["id"], label, prev_score, res["score"], d))

            if improvements:
                r("  Improved vs previous:")
                for fid, lbl, ps, cs, d in sorted(improvements, key=lambda x: -x[4]):
                    r(f"    {fid}  {lbl:35s}  {ps:.2f} → {cs:.2f}  ▲{d:.2f}")
            if regressions:
                r("  Regressed vs previous:")
                for fid, lbl, ps, cs, d in sorted(regressions, key=lambda x: x[4]):
                    r(f"    {fid}  {lbl:35s}  {ps:.2f} → {cs:.2f}  ▼{abs(d):.2f}")
            if not improvements and not regressions:
                r("  No meaningful score changes vs previous run (all deltas < 0.05)")

    r()
    r("  DIAGNOSIS:")
    if ft_data and base_data:
        ft_avg   = ft_data["summary"]["avg_score"]
        base_avg = base_data["summary"]["avg_score"]
        delta    = ft_avg - base_avg
        if ft_avg < base_avg - 0.05:
            r("  → Fine-tuning is HURTING the model.")
            r("    Check: data format (ShareGPT 'conversations' key?), lower LR, fewer steps.")
        elif delta < 0.05:
            r("  → Fine-tuning has no measurable effect over base.")
            r("    Check: is TRAIN_DATA being read? Is the adapter loading in vLLM?")
            r(f"    Verify vLLM is serving '{LORA_MODEL}', not just the base model.")
        elif ft_avg < 0.35:
            r(f"  → Training is helping (+{delta:.2f}) but scores are still very low.")
            r("    1. Check max_steps in config.yaml — need enough steps for your dataset size.")
            r("    2. Once steps are right, add more data for the weak conventions above.")
        elif ft_avg < 0.6:
            r("  → Training is working. Focus on the weak conventions above.")
        else:
            r("  → Good scores. Validate on real tasks.")
    elif ft_data:
        ft_avg = ft_data["summary"]["avg_score"]
        if ft_avg < 0.35:
            r("  → Scores are low. Check max_steps and add data for weak conventions.")
        elif ft_avg < 0.6:
            r("  → Training working. Add data for weak conventions above.")
        else:
            r("  → Good scores. Validate on real tasks.")

    if ft_data and ft_data["summary"]["avg_score"] >= 0.8:
        r("  → Holdout scores strong. Next step: real task evaluation.")

    r()
    r("  Eval reports:")
    if ft_path:
        r(f"    Fine-tuned: {ft_path.with_suffix('.md')}")
    if base_path:
        r(f"    Base:       {base_path.with_suffix('.md')}")
    if status_path:
        r(f"    Synth status (for reposynth): {status_path}")
    r()
    r(f"  Full cycle log: {LOG_PATH}")
    r("=" * 64)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip-train",     action="store_true", help="Skip training step")
    p.add_argument("--skip-serve",     action="store_true", help="Skip starting vLLM")
    p.add_argument("--skip-eval",      action="store_true", help="Skip evaluation steps")
    p.add_argument("--skip-base-eval", action="store_true",
                   help="Skip base model eval (saves time if you only want fine-tuned scores)")
    p.add_argument("--version", metavar="VER",
                   help=(
                       "Data version tag, e.g. --version 20260416. "
                       "Expects <data_dir>/training_VER.jsonl and holdout_VER.jsonl. "
                       "Omit to use data.train / data.holdout from config.yaml."
                   ))
    p.add_argument("--canary", action="store_true",
                   help="Quick validation: train for 300 steps instead of max_steps (~35 min vs full run).")
    p.add_argument("--steps", type=int, metavar="N",
                   help="Override max_steps from config.yaml (ignored if --skip-train).")
    p.add_argument("--auto-steps", action="store_true",
                   help=(
                       "Compute max_steps as 4 × records / effective_batch (≈4 epochs). "
                       "Ignored if --steps or --canary is also set."
                   ))
    p.add_argument("--no-best-checkpoint", action="store_true",
                   help=(
                       "Skip best-checkpoint selection. By default, cycle.py evaluates every "
                       "saved checkpoint after training and promotes the highest-scoring one "
                       "to final/. Use this flag to keep the last-step adapter unconditionally."
                   ))
    p.add_argument("--holdout", metavar="PATH",
                   help="Override holdout file path for eval.")
    return p.parse_args()


def main() -> None:
    global TRAIN_DATA, HOLDOUT

    args = parse_args()
    _setup_logging()

    if args.version:
        TRAIN_DATA = DATA_DIR / f"training_{args.version}.jsonl"
        HOLDOUT    = DATA_DIR / f"holdout_{args.version}.jsonl"

    if args.holdout:
        HOLDOUT = Path(args.holdout).expanduser()
        log(f"Holdout override: {HOLDOUT}")

    log(f"cycle.py started — log: {LOG_PATH}")
    log(f"Model:      {BASE_MODEL}")
    log(f"Adapter:    {LORA_MODEL}")
    log(f"Train data: {TRAIN_DATA}")
    log(f"Holdout:    {HOLDOUT}")
    log(f"Flags: skip_train={args.skip_train} skip_serve={args.skip_serve} "
        f"skip_eval={args.skip_eval} skip_base_eval={args.skip_base_eval} "
        f"canary={args.canary} steps={args.steps} auto_steps={args.auto_steps} "
        f"no_best_checkpoint={args.no_best_checkpoint}")

    if not args.skip_eval and not HOLDOUT.exists():
        die(f"Holdout file not found: {HOLDOUT}\n  Fix: place your holdout JSONL at that path, or use --holdout.")

    vllm_proc   = None
    cycle_start = time.time()

    try:
        if not args.skip_train:
            t0 = time.time()
            step_backup()
            log(f"Backup step completed in {_elapsed(t0)}")

        if not args.skip_train:
            t0 = time.time()
            step_stop_vllm()
            log(f"Stop vLLM step completed in {_elapsed(t0)}")

        if not args.skip_train:
            max_steps_override: int | None = None
            if args.canary:
                max_steps_override = 300
                log("Canary mode: max_steps=300")
            elif args.steps is not None:
                max_steps_override = args.steps
            t0 = time.time()
            step_train(max_steps=max_steps_override,
                       auto_steps=args.auto_steps and not args.canary)
            log(f"Training step completed in {_elapsed(t0)}")

        do_best_checkpoint = not args.no_best_checkpoint and not args.skip_train

        if not args.skip_serve:
            t0 = time.time()
            vllm_proc = step_start_vllm(include_checkpoints=do_best_checkpoint)
            log(f"vLLM startup completed in {_elapsed(t0)}")
        else:
            log("Skipping vLLM start (--skip-serve) — assuming server is already running")

        ft_path     = None
        base_path   = None
        status_path = None

        if not args.skip_eval:
            t0 = time.time()
            if do_best_checkpoint:
                ft_path = step_select_best_checkpoint()
            else:
                ft_path = step_eval(LORA_MODEL, "fine-tuned")
            log(f"Fine-tuned eval completed in {_elapsed(t0)}")

            if not args.skip_base_eval:
                t0 = time.time()
                base_path = step_eval(BASE_MODEL, "base-model")
                log(f"Base eval completed in {_elapsed(t0)}")
            else:
                log("Skipping base model eval (--skip-base-eval)")

            status_path = step_emit_synth_status(ft_path)

        step_report(ft_path, base_path, status_path)

    except KeyboardInterrupt:
        log("Interrupted by user", "WARN")
    finally:
        if vllm_proc and vllm_proc.poll() is None:
            log("Leaving vLLM server running.")
            log(f"To stop: kill {vllm_proc.pid}  or  pkill -f 'vllm serve'")

    log(f"\nCycle complete in {_elapsed(cycle_start)}. Log: {LOG_PATH}")


if __name__ == "__main__":
    main()
