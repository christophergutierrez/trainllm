"""Endpoints for triggering pipeline runs."""

import asyncio
import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..config import cfg
from ..ws import manager, Channel

router = APIRouter()

_active_process: subprocess.Popen | None = None


@router.post("/start")
async def start_cycle(skip_train: bool = False, skip_judge: bool = False):
    """Start a cycle.py run as a subprocess."""
    global _active_process
    if _active_process and _active_process.poll() is None:
        raise HTTPException(409, "A cycle is already running")

    cycle_script = cfg.base_dir / "cycle.py"
    if not cycle_script.exists():
        raise HTTPException(404, "cycle.py not found")

    cmd = [sys.executable, str(cycle_script)]
    if skip_train:
        cmd.append("--skip-train")
    if skip_judge:
        cmd.append("--skip-judge")

    _active_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(cfg.base_dir),
    )

    asyncio.get_event_loop().create_task(_stream_output(_active_process))

    return {"status": "started", "pid": _active_process.pid}


@router.get("/status")
async def cycle_status():
    """Check if a cycle is currently running."""
    if _active_process is None:
        return {"running": False}
    poll = _active_process.poll()
    if poll is None:
        return {"running": True, "pid": _active_process.pid}
    return {"running": False, "exit_code": poll}


@router.post("/stop")
async def stop_cycle():
    """Stop a running cycle."""
    global _active_process
    if _active_process is None or _active_process.poll() is not None:
        raise HTTPException(404, "No cycle is running")
    _active_process.terminate()
    return {"status": "terminated"}


async def _stream_output(proc: subprocess.Popen):
    """Stream subprocess output to WebSocket clients."""
    loop = asyncio.get_event_loop()
    while True:
        line = await loop.run_in_executor(None, proc.stdout.readline)
        if not line and proc.poll() is not None:
            break
        if line:
            await manager.broadcast(Channel.TRAINING, {
                "type": "log",
                "content": line.rstrip(),
            })
    await manager.broadcast(Channel.TRAINING, {
        "type": "cycle_end",
        "exit_code": proc.returncode,
    })
