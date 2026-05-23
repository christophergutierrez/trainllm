"""Single endpoint for the Training page — all metrics computed server-side."""

import json
from datetime import datetime, timezone

from fastapi import APIRouter

from ..config import cfg
from .. import charts

router = APIRouter()


def _read_events() -> list[dict]:
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


def _latest_convergence() -> dict | None:
    from .runs import _scan_runs, _find_convergence
    runs = _scan_runs()
    if not runs:
        return None
    run_id = runs[0]["id"]
    conv = _find_convergence(run_id)
    eval_path = cfg.evals_dir / f"{run_id}.json"
    eval_score = None
    if eval_path.exists():
        try:
            data = json.loads(eval_path.read_text())
            eval_score = data.get("summary", {}).get("avg_score")
        except (json.JSONDecodeError, KeyError):
            pass
    return {"convergence": conv, "eval_score": eval_score, "run_id": run_id}


@router.get("")
async def training_state():
    events = _read_events()
    loss_events = [e for e in events if e.get("event") == "loss"]
    errors = [e for e in events if e.get("event") == "error"]
    warnings = [e for e in events if e.get("event") == "warning"]

    # Determine status
    status = "idle"
    if events:
        last = events[-1]
        if last.get("event") == "step_end" and last.get("step") == "train":
            status = "complete"
        elif last.get("event") == "error":
            status = "error"
        elif any(e.get("event") == "loss" for e in events[-5:]):
            status = "training"
        elif any(e.get("event") == "step_start" for e in events[-3:]):
            status = "starting"

    # Live metrics from events
    latest = loss_events[-1] if loss_events else {}
    step = latest.get("step", 0)
    loss = latest.get("value")
    lr = latest.get("lr")
    epoch = latest.get("epoch")

    # s/step and elapsed from loss timestamps
    sec_per_step = None
    elapsed_sec = None
    if len(loss_events) >= 2:
        try:
            first_ts = datetime.fromisoformat(loss_events[0]["timestamp"])
            last_ts = datetime.fromisoformat(loss_events[-1]["timestamp"])
            elapsed_sec = round((last_ts - first_ts).total_seconds(), 1)

            prev_ts = datetime.fromisoformat(loss_events[-2]["timestamp"])
            dt = (last_ts - prev_ts).total_seconds()
            ds = loss_events[-1]["step"] - loss_events[-2]["step"]
            if ds > 0:
                sec_per_step = round(dt / ds, 1)
        except (KeyError, ValueError):
            pass

    # max_steps from config or events
    max_steps = 0
    for e in events:
        if e.get("event") == "step_start" and e.get("max_steps"):
            max_steps = e["max_steps"]
    if not max_steps:
        try:
            import yaml
            pipeline_cfg = yaml.safe_load((cfg.base_dir / "config.yaml").read_text()) or {}
            max_steps = pipeline_cfg.get("training", {}).get("max_steps", 0)
        except Exception:
            pass

    progress = round(step / max_steps * 100) if max_steps > 0 and step > 0 else None
    remaining_sec = round((max_steps - step) * sec_per_step) if sec_per_step and max_steps > step else None

    # Fall back to last completed run when idle
    hist_loss = None
    hist_step = None
    hist_lr = None
    hist_eval_score = None
    hist_run_id = None
    if status == "idle":
        hist = _latest_convergence()
        if hist and hist["convergence"]:
            conv = hist["convergence"]
            hist_loss = conv.get("final_loss")
            hist_step = conv.get("total_steps")
            hist_lr = conv.get("final_lr")
            hist_eval_score = hist.get("eval_score")
            hist_run_id = hist.get("run_id")
            if not max_steps:
                max_steps = hist_step or 0

    # Build chart server-side
    chart = None
    if loss_events and len(loss_events) >= 2:
        chart = charts.loss_curve(
            [e["step"] for e in loss_events],
            [e["value"] for e in loss_events],
        )
    elif status == "idle" and hist_run_id:
        try:
            from .runs import _find_convergence
            conv = _find_convergence(hist_run_id)
            if conv and conv.get("loss_history"):
                h = conv["loss_history"]
                chart = charts.loss_curve([p[0] for p in h], [p[1] for p in h])
        except Exception:
            pass

    return {
        "status": status,
        "step": step or hist_step or 0,
        "loss": loss if loss is not None else hist_loss,
        "lr": lr if lr is not None else hist_lr,
        "eval_score": hist_eval_score,
        "epoch": epoch,
        "max_steps": max_steps,
        "progress_pct": progress,
        "sec_per_step": sec_per_step,
        "elapsed_sec": elapsed_sec,
        "remaining_sec": remaining_sec,
        "errors": errors[-5:],
        "warnings": warnings[-5:],
        "source": "live" if loss_events else "historical",
        "hist_run_id": hist_run_id,
        "chart": chart,
    }
