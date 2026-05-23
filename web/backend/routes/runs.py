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

        # Build search order: most recent final-* dir first, then final, then .
        versioned = sorted(
            adapter_dir.glob("final-*"),
            key=lambda p: p.name,
            reverse=True,
        )
        subdirs = [v.name for v in versioned] + ["final", "."]

        # Prefer trainer_state.json from versioned/final dirs (freshest data)
        parsed = _extract_from_subdirs(adapter_dir, subdirs)
        if parsed:
            return parsed

        # Fall back to convergence.json files
        for subdir in subdirs:
            conv_path = adapter_dir / subdir / "convergence.json"
            if conv_path.exists():
                try:
                    return json.loads(conv_path.read_text())
                except json.JSONDecodeError:
                    pass

    return None


def _extract_from_subdirs(lora_dir: Path, subdirs: list[str]) -> dict | None:
    """Extract convergence data from trainer_state.json, searching subdirs in priority order."""
    for subdir in subdirs:
        target = lora_dir / subdir
        if not target.is_dir():
            continue
        for state_file in _find_trainer_states(target):
            result = _parse_trainer_state(state_file)
            if result:
                return result
    return None


def _find_trainer_states(directory: Path):
    direct = directory / "trainer_state.json"
    if direct.exists():
        yield direct
    for ckpt in sorted(directory.glob("checkpoint-*"), key=lambda p: p.name, reverse=True):
        f = ckpt / "trainer_state.json"
        if f.exists():
            yield f


def _parse_trainer_state(state_file: Path) -> dict | None:
    try:
        state = json.loads(state_file.read_text())
        log_history = state.get("log_history", [])
        loss_entries = [e for e in log_history if "loss" in e]
        if not loss_entries:
            return None
        last = loss_entries[-1]
        return {
            "first_loss": loss_entries[0]["loss"],
            "final_loss": last["loss"],
            "total_steps": last["step"],
            "final_lr": last.get("learning_rate"),
            "final_epoch": last.get("epoch"),
            "loss_history": [[e["step"], e["loss"]] for e in loss_entries],
        }
    except (json.JSONDecodeError, KeyError):
        return None
