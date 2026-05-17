"""Endpoints for training run history."""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..config import cfg
from .. import charts

router = APIRouter()


def _scan_runs() -> list[dict]:
    """Scan evals directory to build run history."""
    runs = []
    if not cfg.evals_dir.exists():
        return runs

    seen = set()
    for f in sorted(cfg.evals_dir.glob("*.json"), reverse=True):
        if "_llmjudge" in f.name or "_synth" in f.name:
            continue
        run_id = f.stem
        if run_id in seen:
            continue
        seen.add(run_id)

        try:
            data = json.loads(f.read_text())
            summary = data.get("summary", {})
            meta = data.get("meta", {})
            runs.append({
                "id": run_id,
                "file": f.name,
                "timestamp": meta.get("timestamp", run_id[:16]),
                "model": meta.get("model", ""),
                "avg_score": summary.get("avg_score", 0),
                "avg_composite_score": summary.get("avg_composite_score"),
                "avg_structural_score": summary.get("avg_structural_score"),
                "band_counts": summary.get("band_counts", {}),
                "num_records": len(data.get("results", [])),
            })
        except (json.JSONDecodeError, KeyError):
            continue

    return runs


@router.get("")
async def list_runs():
    return _scan_runs()


@router.get("/{run_id}")
async def get_run(run_id: str):
    eval_path = cfg.evals_dir / f"{run_id}.json"
    if not eval_path.exists():
        raise HTTPException(404, f"Run not found: {run_id}")
    data = json.loads(eval_path.read_text())

    convergence = _find_convergence(run_id)
    judge_path = cfg.evals_dir / f"{run_id}_llmjudge.json"
    judge_data = None
    if judge_path.exists():
        judge_data = json.loads(judge_path.read_text())

    return {
        "eval": data,
        "convergence": convergence,
        "judge": judge_data,
    }


@router.get("/{run_id}/loss-chart")
async def run_loss_chart(run_id: str):
    convergence = _find_convergence(run_id)
    if not convergence:
        raise HTTPException(404, "No convergence data for this run")

    events = convergence.get("loss_history", [])
    if not events:
        raise HTTPException(404, "No loss history in convergence data")

    steps = [e[0] if isinstance(e, list) else e["step"] for e in events]
    losses = [e[1] if isinstance(e, list) else e["loss"] for e in events]
    return charts.loss_curve(steps, losses)


def _find_convergence(run_id: str) -> dict | None:
    """Find convergence.json for a run based on the model name in the eval data."""
    import re
    # Extract adapter/model name from the eval metadata
    eval_path = cfg.evals_dir / f"{run_id}.json"
    model_name = cfg.adapter_name
    if eval_path.exists():
        try:
            meta = json.loads(eval_path.read_text()).get("meta", {})
            model_name = meta.get("model", model_name)
        except (json.JSONDecodeError, KeyError):
            pass

    # Strip checkpoint suffixes like "-ckpt1200" to get the base adapter name
    base_adapter = re.sub(r"-ckpt\d+$", "", model_name)

    # Try adapter-specific lora directory (most specific first)
    candidates = [
        cfg.lora_dir / model_name,
        cfg.lora_dir / base_adapter,
        cfg.lora_dir / cfg.adapter_name,
    ]

    for adapter_dir in candidates:
        if not adapter_dir.exists():
            continue

        conv_data = None
        for subdir in [".", "final"]:
            conv_path = adapter_dir / subdir / "convergence.json"
            if conv_path.exists():
                try:
                    conv_data = json.loads(conv_path.read_text())
                    break
                except json.JSONDecodeError:
                    pass

        if conv_data:
            if "loss_history" not in conv_data:
                loss_history = _extract_loss_history(adapter_dir)
                if loss_history:
                    conv_data["loss_history"] = loss_history
            return conv_data

    return None


def _extract_loss_history(lora_dir: Path) -> list:
    """Extract loss history from the latest trainer_state.json checkpoint."""
    checkpoints = sorted(lora_dir.glob("checkpoint-*"), key=lambda p: p.name)
    if not checkpoints:
        return []
    state_file = checkpoints[-1] / "trainer_state.json"
    if not state_file.exists():
        return []
    try:
        state = json.loads(state_file.read_text())
        log_history = state.get("log_history", [])
        return [
            [entry["step"], entry["loss"]]
            for entry in log_history
            if "loss" in entry
        ]
    except (json.JSONDecodeError, KeyError):
        return []
