"""Endpoints for model management, diagnostics, comparison, and download."""

import json
import tarfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from ..config import cfg
from .. import charts

router = APIRouter()

MODEL_DIR = cfg.base_dir / "models"
REGISTRY_PATH = cfg.base_dir / "models.json"


def _read_registry() -> dict:
    if not REGISTRY_PATH.exists():
        return {"models": []}
    data = json.loads(REGISTRY_PATH.read_text())
    if "bundles" in data and "models" not in data:
        data["models"] = data.pop("bundles")
    return data


def _extract_from_tarball(model_name: str, member_path: str) -> Any | None:
    """Extract and parse a JSON file from inside a model tarball."""
    tarball = MODEL_DIR / f"{model_name}.tar.gz"
    if not tarball.exists():
        return None
    try:
        with tarfile.open(tarball, "r:gz") as tar:
            f = tar.extractfile(f"{model_name}/{member_path}")
            if f is None:
                return None
            return json.loads(f.read())
    except (KeyError, json.JSONDecodeError):
        return None


def _extract_model_data(model_name: str) -> dict:
    """Extract all comparison-relevant data from a model tarball."""
    manifest = _extract_from_tarball(model_name, "manifest.json")
    if not manifest:
        raise HTTPException(404, f"Model not found: {model_name}")

    eval_data = _extract_from_tarball(model_name, "diagnostics/eval.json")
    convergence = _extract_from_tarball(model_name, "diagnostics/convergence.json")
    loss_history = _extract_from_tarball(model_name, "diagnostics/loss_history.json")

    convention_breakdown = []
    if eval_data:
        summary = eval_data.get("summary", {})
        convention_breakdown = summary.get("convention_breakdown", [])

    eval_info = manifest.get("eval") or {}

    return {
        "model_name": manifest.get("model_name") or manifest.get("bundle_name", model_name),
        "adapter": manifest.get("adapter", ""),
        "version": manifest.get("version", 0),
        "score": eval_info.get("score"),
        "base_model": manifest.get("base_model", ""),
        "created": manifest.get("created", ""),
        "band_counts": eval_info.get("band_counts", {}),
        "num_records": eval_info.get("num_records", 0),
        "convergence": convergence,
        "loss_history": loss_history,
        "convention_breakdown": convention_breakdown,
        "lora": manifest.get("lora"),
        "git_sha": manifest.get("git_sha"),
        "config_hash": manifest.get("config_hash"),
        "train_data": manifest.get("train_data"),
    }


@router.get("")
async def list_models():
    """List all model artifacts from the registry."""
    return _read_registry()


@router.get("/compare")
async def compare_models(names: str = Query(..., description="Comma-separated model names")):
    """Compare multiple models — returns merged metrics and chart specs."""
    name_list = [n.strip() for n in names.split(",") if n.strip()]
    if len(name_list) < 2:
        raise HTTPException(400, "Need at least 2 model names to compare")

    models_data = []
    for name in name_list:
        try:
            models_data.append(_extract_model_data(name))
        except HTTPException:
            raise HTTPException(404, f"Model not found: {name}")

    comparison_charts = {}

    has_loss = any(m.get("loss_history") for m in models_data)
    if has_loss:
        comparison_charts["loss_overlay"] = charts.loss_comparison(models_data)

    comparison_charts["score_comparison"] = charts.score_comparison(models_data)

    has_bands = any(m.get("band_counts") for m in models_data)
    if has_bands:
        comparison_charts["band_comparison"] = charts.band_comparison(models_data)

    has_conventions = any(m.get("convention_breakdown") for m in models_data)
    if has_conventions:
        comparison_charts["convention_comparison"] = charts.convention_comparison(models_data)

    return {"models": models_data, "charts": comparison_charts}


@router.get("/{model_name}")
async def get_model_manifest(model_name: str):
    """Get the manifest from a specific model."""
    manifest = _extract_from_tarball(model_name, "manifest.json")
    if not manifest:
        raise HTTPException(404, f"Model not found: {model_name}")
    return manifest


@router.get("/{model_name}/diagnostics")
async def get_model_diagnostics(model_name: str):
    """Get full diagnostics for a single model."""
    return _extract_model_data(model_name)


@router.get("/{model_name}/download")
async def download_model(model_name: str):
    """Download a model tarball."""
    tarball = MODEL_DIR / f"{model_name}.tar.gz"
    if not tarball.exists():
        raise HTTPException(404, f"Model not found: {model_name}")
    return FileResponse(
        path=str(tarball),
        media_type="application/gzip",
        filename=f"{model_name}.tar.gz",
    )
