#!/usr/bin/env python3
"""
Bundle a trained LoRA adapter into a versioned, deployable model artifact.

Creates a tarball containing everything needed to serve the adapter (weights,
tokenizer, config) plus a manifest with full provenance (model, scores, git SHA,
training config snapshot) and diagnostics for comparison (eval results, loss
history, convergence metrics).

Usage:
    python bundle.py                                   # bundle current adapter from config
    python bundle.py --adapter-dir lora/api-thinking/final
    python bundle.py --adapter-dir lora/api-thinking/final --min-score 0.95
    python bundle.py --eval-id 2026-05-15_131650_api-thinking  # specific eval

Output:
    models/{adapter}-v{N}.tar.gz
    Updates models.json registry
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import _config

MODEL_DIR = Path(__file__).parent / "models"
REGISTRY_PATH = Path(__file__).parent / "models.json"

INFERENCE_FILES = {
    "adapter_model.safetensors",
    "adapter_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
}

OPTIONAL_FILES = {
    "special_tokens_map.json",
    "added_tokens.json",
}


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent, text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _git_dirty() -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).parent, capture_output=True, text=True,
        )
        return bool(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _config_hash(cfg_path: Path) -> str:
    if not cfg_path.exists():
        return "unknown"
    return hashlib.sha256(cfg_path.read_bytes()).hexdigest()[:12]


def _find_best_eval(evals_dir: Path, adapter_name: str) -> dict | None:
    """Find the highest-scoring eval for this adapter."""
    best = None
    best_score = -1
    for f in evals_dir.glob("*.json"):
        if "_llmjudge" in f.name or "_synth" in f.name:
            continue
        if adapter_name not in f.stem:
            continue
        try:
            data = json.loads(f.read_text())
            summary = data.get("summary", {})
            score = summary.get("avg_composite_score") or summary.get("avg_score", 0)
            if score > best_score:
                best_score = score
                best = {
                    "eval_id": f.stem,
                    "score": round(score, 4),
                    "band_counts": summary.get("band_counts", {}),
                    "num_records": len(data.get("results", [])),
                }
        except (json.JSONDecodeError, KeyError):
            continue
    return best


def _load_eval(evals_dir: Path, eval_id: str) -> dict | None:
    f = evals_dir / f"{eval_id}.json"
    if not f.exists():
        return None
    data = json.loads(f.read_text())
    summary = data.get("summary", {})
    score = summary.get("avg_composite_score") or summary.get("avg_score", 0)
    return {
        "eval_id": eval_id,
        "score": round(score, 4),
        "band_counts": summary.get("band_counts", {}),
        "num_records": len(data.get("results", [])),
    }


def _next_version(adapter_name: str) -> int:
    """Get next version number from registry."""
    if not REGISTRY_PATH.exists():
        return 1
    registry = json.loads(REGISTRY_PATH.read_text())
    versions = [
        m["version"] for m in registry.get("models", [])
        if m["adapter"] == adapter_name
    ]
    return max(versions, default=0) + 1


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract_loss_history(adapter_dir: Path) -> list[list] | None:
    """Extract [step, loss] pairs from the latest checkpoint's trainer_state."""
    parent = adapter_dir.parent if adapter_dir.name == "final" else adapter_dir
    checkpoints = sorted(
        [d for d in parent.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda p: int(p.name.split("-")[1]) if p.name.split("-")[1].isdigit() else 0,
    )
    if not checkpoints:
        return None
    ts_path = checkpoints[-1] / "trainer_state.json"
    if not ts_path.exists():
        return None
    ts = json.loads(ts_path.read_text())
    history = []
    for entry in ts.get("log_history", []):
        if "loss" in entry and "step" in entry:
            history.append([entry["step"], round(entry["loss"], 6)])
    return history if history else None


def bundle(
    adapter_dir: Path,
    adapter_name: str,
    eval_info: dict | None,
    min_score: float,
    cfg_path: Path,
    train_data_path: Path | None = None,
    evals_dir: Path | None = None,
) -> Path | None:
    """Create a versioned model tarball. Returns path or None if gate fails."""

    # Score gate
    if eval_info:
        if eval_info["score"] < min_score:
            print(f"REJECTED: score {eval_info['score']:.4f} < min {min_score:.4f}")
            return None
    else:
        print("WARNING: No eval found. Bundling without score gate.")

    # Check required files exist
    missing = []
    for fname in INFERENCE_FILES:
        if not (adapter_dir / fname).exists():
            missing.append(fname)
    if missing:
        print(f"ERROR: Missing required files in {adapter_dir}: {missing}")
        return None

    version = _next_version(adapter_name)
    model_name = f"{adapter_name}-v{version}"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODEL_DIR / f"{model_name}.tar.gz"

    # Read adapter config for base model info
    adapter_cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    base_model = adapter_cfg.get("base_model_name_or_path", "unknown")

    # Training data provenance
    train_data_info = None
    if train_data_path and train_data_path.exists():
        train_data_info = {
            "path": str(train_data_path),
            "sha256": _sha256_file(train_data_path),
            "size": train_data_path.stat().st_size,
        }

    # Build manifest
    manifest = {
        "schema_version": 3,
        "model_name": model_name,
        "adapter": adapter_name,
        "version": version,
        "base_model": base_model,
        "created": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "config_hash": _config_hash(cfg_path),
        "train_data": train_data_info,
        "eval": eval_info,
        "lora": {
            "rank": adapter_cfg.get("r"),
            "alpha": adapter_cfg.get("lora_alpha"),
            "target_modules": adapter_cfg.get("target_modules"),
            "peft_type": adapter_cfg.get("peft_type"),
        },
        "files": {},
        "diagnostics": {
            "has_eval": False,
            "has_convergence": False,
            "has_loss_history": False,
        },
    }

    # Create tarball
    with tempfile.TemporaryDirectory() as tmp:
        manifest_path = Path(tmp) / "manifest.json"

        # Collect inference files
        files_to_add = []
        for fname in INFERENCE_FILES | OPTIONAL_FILES:
            src = adapter_dir / fname
            if src.exists():
                files_to_add.append((src, fname))
                manifest["files"][fname] = {
                    "size": src.stat().st_size,
                    "sha256": _sha256_file(src),
                }

        # Include config.yaml snapshot
        if cfg_path.exists():
            files_to_add.append((cfg_path, "config.yaml"))
            manifest["files"]["config.yaml"] = {
                "size": cfg_path.stat().st_size,
                "sha256": _sha256_file(cfg_path),
            }

        # Collect diagnostics
        diagnostics_files = []

        # Full eval results
        if eval_info and evals_dir:
            eval_file = evals_dir / f"{eval_info['eval_id']}.json"
            if eval_file.exists():
                diagnostics_files.append((eval_file, "diagnostics/eval.json"))
                manifest["diagnostics"]["has_eval"] = True

        # Convergence metrics
        for conv_path in [adapter_dir / "convergence.json", adapter_dir.parent / "convergence.json"]:
            if conv_path.exists():
                diagnostics_files.append((conv_path, "diagnostics/convergence.json"))
                manifest["diagnostics"]["has_convergence"] = True
                break

        # Loss history
        loss_history = _extract_loss_history(adapter_dir)
        if loss_history:
            loss_path = Path(tmp) / "loss_history.json"
            loss_path.write_text(json.dumps(loss_history))
            diagnostics_files.append((loss_path, "diagnostics/loss_history.json"))
            manifest["diagnostics"]["has_loss_history"] = True

        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

        with tarfile.open(out_path, "w:gz") as tar:
            tar.add(str(manifest_path), arcname=f"{model_name}/manifest.json")
            for src, arcname in files_to_add:
                tar.add(str(src), arcname=f"{model_name}/{arcname}")
            for src, arcname in diagnostics_files:
                tar.add(str(src), arcname=f"{model_name}/{arcname}")

    # Update registry
    _update_registry(manifest, out_path)

    return out_path


def _update_registry(manifest: dict, tarball_path: Path):
    """Add model to models.json registry."""
    if REGISTRY_PATH.exists():
        registry = json.loads(REGISTRY_PATH.read_text())
    else:
        registry = {"models": []}

    # Handle legacy "bundles" key
    if "bundles" in registry and "models" not in registry:
        registry["models"] = registry.pop("bundles")

    entry = {
        "adapter": manifest["adapter"],
        "version": manifest["version"],
        "model_name": manifest["model_name"],
        "base_model": manifest["base_model"],
        "score": manifest["eval"]["score"] if manifest["eval"] else None,
        "created": manifest["created"],
        "git_sha": manifest["git_sha"],
        "filename": tarball_path.name,
        "size_mb": round(tarball_path.stat().st_size / 1024 / 1024, 1),
    }
    registry["models"].append(entry)
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2) + "\n")


def main():
    cfg = _config.load()

    parser = argparse.ArgumentParser(description="Bundle a LoRA adapter into a versioned model")
    parser.add_argument("--adapter-dir", type=str, default=None,
                        help="Path to adapter directory (default: lora/{adapter}/final)")
    parser.add_argument("--adapter-name", type=str, default=None,
                        help="Adapter name for versioning (default: from config)")
    parser.add_argument("--eval-id", type=str, default=None,
                        help="Specific eval to use for score gate")
    parser.add_argument("--min-score", type=float, default=0.9,
                        help="Minimum composite score to allow bundling")
    parser.add_argument("--force", action="store_true",
                        help="Skip score gate (still records score if available)")
    args = parser.parse_args()

    adapter_name = args.adapter_name or cfg.adapter_name
    adapter_dir = Path(args.adapter_dir) if args.adapter_dir else cfg.lora_dir / "final"
    if not adapter_dir.exists():
        adapter_dir = cfg.base_dir / "lora" / adapter_name / "final"

    env_cfg = os.environ.get("TRAINLLM_CONFIG")
    cfg_path = Path(env_cfg) if env_cfg else Path(__file__).parent / "config.yaml"

    print(f"Adapter:    {adapter_name}")
    print(f"Source:     {adapter_dir}")
    print(f"Min score:  {args.min_score}")

    # Find eval
    if args.eval_id:
        eval_info = _load_eval(cfg.evals_dir, args.eval_id)
    else:
        eval_info = _find_best_eval(cfg.evals_dir, adapter_name)

    if eval_info:
        print(f"Eval:       {eval_info['eval_id']}")
        print(f"Score:      {eval_info['score']}")
        print(f"Bands:      {eval_info['band_counts']}")
    else:
        print("Eval:       NONE FOUND")

    if args.force:
        min_score = 0.0
    else:
        min_score = args.min_score

    # Resolve training data path
    train_data_path = cfg.train_data
    if train_data_path.exists():
        print(f"Train data: {train_data_path}")
    else:
        print(f"Train data: NOT FOUND ({train_data_path})")
        train_data_path = None

    print("=" * 60)

    result = bundle(
        adapter_dir, adapter_name, eval_info, min_score, cfg_path,
        train_data_path, evals_dir=cfg.evals_dir,
    )
    if result:
        print(f"\nModel:    {result}")
        print(f"Size:     {result.stat().st_size / 1024 / 1024:.1f} MB")
        print(f"Registry: {REGISTRY_PATH}")
    else:
        print("\nBundle FAILED (see above)")
        sys.exit(1)


if __name__ == "__main__":
    main()
