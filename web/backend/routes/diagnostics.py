"""Endpoints for system diagnostics and timing data."""

import json
import subprocess
from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..config import cfg
from .. import charts

router = APIRouter()


@router.get("/timing")
async def get_timing():
    """Get step timing from the most recent events file."""
    events = _read_events()
    steps = _extract_step_timing(events)
    return {"steps": steps}


@router.get("/timing/chart")
async def timing_chart():
    """Get timing bar chart."""
    events = _read_events()
    steps = _extract_step_timing(events)
    if not steps:
        raise HTTPException(404, "No timing data available")
    names = [s["step"] for s in steps]
    durations = [s["duration_sec"] for s in steps]
    return charts.timing_bars(names, durations)


@router.get("/convergence")
async def get_convergence():
    """Get the most recent convergence.json."""
    adapter = cfg.adapter_name
    paths_to_try = [
        cfg.lora_dir / adapter / "convergence.json",
        cfg.lora_dir / adapter / "final" / "convergence.json",
    ]
    for p in paths_to_try:
        if p.exists():
            return json.loads(p.read_text())

    # Try any convergence.json in lora dir
    for conv in cfg.lora_dir.rglob("convergence.json"):
        return json.loads(conv.read_text())

    raise HTTPException(404, "No convergence data found")


def _safe_int(v: str) -> int | None:
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


@router.get("/gpu")
async def gpu_stats():
    """Return current GPU utilization and system memory (for unified-memory GPUs)."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,name,temperature.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            raise HTTPException(503, "nvidia-smi failed")
        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            gpu_util = _safe_int(parts[0])
            name = parts[1]
            temp = _safe_int(parts[2])
            mem_used = _safe_int(parts[3])
            mem_total = _safe_int(parts[4])

            gpu: dict = {"name": name}
            if gpu_util is not None:
                gpu["utilization_pct"] = gpu_util
            if temp is not None:
                gpu["temperature_c"] = temp
            if mem_used is not None and mem_total is not None and mem_total > 0:
                gpu["memory_used_mb"] = mem_used
                gpu["memory_total_mb"] = mem_total
                gpu["memory_pct"] = round(mem_used / mem_total * 100, 1)
            gpus.append(gpu)

        if gpus and "memory_used_mb" not in gpus[0]:
            try:
                mem = subprocess.run(
                    ["free", "-m"],
                    capture_output=True, text=True, timeout=5,
                )
                for mline in mem.stdout.splitlines():
                    if mline.startswith("Mem:"):
                        mparts = mline.split()
                        total = int(mparts[1])
                        used = int(mparts[2])
                        gpus[0]["memory_used_mb"] = used
                        gpus[0]["memory_total_mb"] = total
                        gpus[0]["memory_pct"] = round(used / total * 100, 1)
                        gpus[0]["unified_memory"] = True
                        break
            except Exception:
                pass

        return {"gpus": gpus}
    except FileNotFoundError:
        raise HTTPException(503, "nvidia-smi not found")
    except subprocess.TimeoutExpired:
        raise HTTPException(503, "nvidia-smi timed out")


@router.get("/data-coverage")
async def data_coverage():
    """Report training data statistics per convention/endpoint."""
    train_data = cfg.base_dir / "data" / "training.jsonl"
    if not train_data.exists():
        raise HTTPException(404, "No training data found")

    conventions: dict[str, int] = {}
    total = 0
    with open(train_data) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                record = json.loads(line)
                for conv in record.get("conversations", []):
                    if conv.get("from") == "human":
                        content = conv.get("value", "")
                        # Extract convention hints from the question
                        # This is a heuristic — works for API training data
                        conventions["total"] = conventions.get("total", 0) + 1
                        break
            except json.JSONDecodeError:
                pass

    return {"total_records": total, "conventions": conventions}


def _read_events() -> list[dict]:
    """Read all events from the events file."""
    if not cfg.events_file.exists():
        return []
    events = []
    with open(cfg.events_file) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return events


def _extract_step_timing(events: list[dict]) -> list[dict]:
    """Extract step start/end pairs into timing records."""
    starts: dict[str, str] = {}
    steps = []
    for e in events:
        etype = e.get("event")
        step_name = e.get("step", "")
        if etype == "step_start":
            starts[step_name] = e.get("timestamp", "")
        elif etype == "step_end" and step_name in starts:
            duration = e.get("duration_sec", 0)
            steps.append({
                "step": step_name,
                "start": starts[step_name],
                "end": e.get("timestamp", ""),
                "duration_sec": duration,
            })
            del starts[step_name]
    return steps
