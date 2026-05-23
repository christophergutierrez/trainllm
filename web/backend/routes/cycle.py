"""Endpoints for triggering pipeline runs."""

import os
import signal
import subprocess
import sys

from fastapi import APIRouter, HTTPException

from ..config import cfg

router = APIRouter()

_active_pid: int | None = None


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


@router.post("/start")
async def start_cycle(skip_train: bool = False, skip_judge: bool = False):
    """Start a cycle.py run as a detached subprocess."""
    global _active_pid
    if _active_pid and _is_pid_alive(_active_pid):
        raise HTTPException(409, "A cycle is already running")

    cycle_script = cfg.base_dir / "cycle.py"
    if not cycle_script.exists():
        raise HTTPException(404, "cycle.py not found")

    cmd = [sys.executable, str(cycle_script)]
    if skip_train:
        cmd.append("--skip-train")
    if skip_judge:
        cmd.append("--skip-judge")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        cwd=str(cfg.base_dir),
    )
    _active_pid = proc.pid

    return {"status": "started", "pid": proc.pid}


@router.get("/status")
async def cycle_status():
    """Check if a cycle is currently running."""
    global _active_pid
    if _active_pid is None:
        return {"running": False}
    if _is_pid_alive(_active_pid):
        return {"running": True, "pid": _active_pid}
    _active_pid = None
    return {"running": False}


@router.post("/stop")
async def stop_cycle():
    """Stop a running cycle."""
    global _active_pid
    if _active_pid is None or not _is_pid_alive(_active_pid):
        raise HTTPException(404, "No cycle is running")
    os.killpg(os.getpgid(_active_pid), signal.SIGTERM)
    _active_pid = None
    return {"status": "terminated"}
