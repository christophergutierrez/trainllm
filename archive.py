#!/usr/bin/env python3
"""Archive adapter weights as versioned tarballs.

Creates a compressed tarball of LoRA adapters (and optionally merged models)
with metadata for later recovery. Use before major changes like swapping the
base model.

Usage:
    python3 archive.py                          # archive all adapters under current base model
    python3 archive.py --adapters my-adapter    # archive specific adapter(s)
    python3 archive.py --include-merged         # also archive merged/ directory
    python3 archive.py --tag nemotron-migration # add a descriptive tag to the archive name
    python3 archive.py --list                   # list existing archives
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import _config

cfg = _config.load()

ARCHIVE_DIR = cfg.base_dir / "archive"
LORA_ROOT = cfg.base_dir / "lora"
MERGED_ROOT = cfg.base_dir / "merged"


def get_adapter_dirs(names: list[str] | None) -> list[Path]:
    if names:
        dirs = []
        for name in names:
            d = LORA_ROOT / name
            if d.exists():
                dirs.append(d)
            else:
                print(f"  WARNING: adapter '{name}' not found at {d}", file=sys.stderr)
        return dirs
    return sorted(d for d in LORA_ROOT.iterdir() if d.is_dir() and d.name != ".gitkeep")


def adapter_summary(adapter_dir: Path) -> dict:
    final = adapter_dir / "final"
    safetensors = list(adapter_dir.rglob("*.safetensors"))
    total_mb = sum(f.stat().st_size for f in safetensors) / 1_048_576

    config_path = final / "adapter_config.json" if final.exists() else None
    lora_rank = None
    if config_path and config_path.exists():
        try:
            lora_rank = json.loads(config_path.read_text()).get("r")
        except Exception:
            pass

    versions = sorted(
        d.name for d in adapter_dir.iterdir()
        if d.is_dir() and d.name.startswith("final-v")
    )

    return {
        "name": adapter_dir.name,
        "lora_rank": lora_rank,
        "total_mb": round(total_mb, 1),
        "has_final": final.exists(),
        "has_dpo": (adapter_dir / "dpo_final").exists(),
        "versions": versions,
        "checkpoint_count": len(list(adapter_dir.glob("checkpoint-*"))),
    }


def git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=str(cfg.base_dir),
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def latest_eval_for(adapter_name: str) -> Path | None:
    safe = adapter_name.replace("/", "_")
    matches = sorted(cfg.evals_dir.glob(f"*_{safe}.json"), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def training_data_stats(adapter_name: str) -> dict | None:
    """Check for per-endpoint training data and return record counts."""
    endpoint = adapter_name.rsplit("-", 1)[-1]
    prepared = cfg.data_dir / "prepared" / endpoint
    stats = {}
    for name in ["training.jsonl", "holdout.jsonl"]:
        f = prepared / name
        if f.exists():
            stats[name.split(".")[0]] = sum(1 for _ in open(f))
    return stats if stats else None


def find_run_config(adapter_name: str) -> Path | None:
    endpoint = adapter_name.rsplit("-", 1)[-1]
    candidates = [
        cfg.data_dir / "runs" / endpoint / "config.yaml",
        cfg.base_dir / "config.yaml",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def create_archive(
    adapter_dirs: list[Path],
    include_merged: bool,
    tag: str | None,
    include_checkpoints: bool,
) -> Path:
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_model_short = cfg.model.replace("/", "_")
    tag_part = f"_{tag}" if tag else ""
    archive_name = f"{base_model_short}{tag_part}_{date_str}"
    tarball_path = ARCHIVE_DIR / f"{archive_name}.tar.gz"

    manifest = {
        "created_at": datetime.now().isoformat() + "Z",
        "base_model": cfg.model,
        "adapter_name": cfg.adapter_name,
        "tag": tag,
        "git_commit": git_commit_hash(),
        "adapters": [],
        "includes_merged": include_merged,
        "includes_checkpoints": include_checkpoints,
    }

    print(f"  Archive: {tarball_path}")
    print(f"  Base model: {cfg.model}")
    print(f"  Adapters: {len(adapter_dirs)}")

    with tarfile.open(tarball_path, "w:gz") as tar:
        for adapter_dir in adapter_dirs:
            summary = adapter_summary(adapter_dir)

            eval_path = latest_eval_for(adapter_dir.name)
            if eval_path:
                summary["eval_file"] = eval_path.name
                try:
                    eval_data = json.loads(eval_path.read_text())
                    summary["eval_avg_score"] = eval_data.get("summary", {}).get("avg_score")
                    summary["eval_date"] = eval_data.get("timestamp")
                except Exception:
                    pass
                tar.add(eval_path, arcname=f"evals/{eval_path.name}")
                md_path = eval_path.with_suffix(".md")
                if md_path.exists():
                    tar.add(md_path, arcname=f"evals/{md_path.name}")

            run_config = find_run_config(adapter_dir.name)
            if run_config:
                summary["training_config"] = run_config.name
                tar.add(run_config, arcname=f"configs/{adapter_dir.name}_config.yaml")

            data_stats = training_data_stats(adapter_dir.name)
            if data_stats:
                summary["data_stats"] = data_stats

            manifest["adapters"].append(summary)

            score_str = ""
            if summary.get("eval_avg_score") is not None:
                score_str = f"  score={summary['eval_avg_score']:.2f}"
            data_str = ""
            if data_stats:
                data_str = f"  train={data_stats.get('training', '?')}"
            print(f"    {summary['name']:30s}  {summary['total_mb']:>8.1f} MB"
                  f"  {'(+DPO)' if summary['has_dpo'] else ''}{score_str}{data_str}")

            for root, dirs, files in os.walk(adapter_dir):
                root_path = Path(root)
                rel = root_path.relative_to(LORA_ROOT)

                if not include_checkpoints and "checkpoint-" in str(rel):
                    continue

                for f in files:
                    filepath = root_path / f
                    arcname = f"lora/{rel}/{f}"
                    tar.add(filepath, arcname=arcname)

        if include_merged and MERGED_ROOT.exists():
            print(f"  Including merged models from {MERGED_ROOT}")
            for root, dirs, files in os.walk(MERGED_ROOT):
                for f in files:
                    filepath = Path(root) / f
                    rel = filepath.relative_to(cfg.base_dir)
                    tar.add(filepath, arcname=str(rel))

        manifest_json = json.dumps(manifest, indent=2)
        import io
        info = tarfile.TarInfo(name="manifest.json")
        data = manifest_json.encode()
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))

    size_mb = tarball_path.stat().st_size / 1_048_576
    print(f"\n  Archive size: {size_mb:.1f} MB")
    print(f"  Path: {tarball_path}")
    return tarball_path


def list_archives():
    if not ARCHIVE_DIR.exists():
        print("No archive directory found.")
        return

    tarballs = sorted(ARCHIVE_DIR.glob("*.tar.gz"))
    if not tarballs:
        print("No archives found.")
        return

    print(f"{'Archive':<60s}  {'Size':>10s}  {'Date':>12s}")
    print("-" * 86)
    for tb in tarballs:
        size_mb = tb.stat().st_size / 1_048_576
        mtime = datetime.fromtimestamp(tb.stat().st_mtime).strftime("%Y-%m-%d")
        print(f"  {tb.name:<58s}  {size_mb:>8.1f} MB  {mtime}")

    print(f"\nTo inspect: tar tzf <archive> | head -20")
    print(f"To extract: tar xzf <archive> -C /target/dir")
    print(f"To read manifest: tar xzf <archive> manifest.json -O | python3 -m json.tool")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapters", help="Comma-separated adapter names (default: all)")
    parser.add_argument("--include-merged", action="store_true", help="Also archive merged/ models")
    parser.add_argument("--include-checkpoints", action="store_true",
                        help="Include intermediate checkpoints (large)")
    parser.add_argument("--tag", help="Descriptive tag for the archive name")
    parser.add_argument("--list", action="store_true", help="List existing archives")
    args = parser.parse_args()

    if args.list:
        list_archives()
        return

    names = [n.strip() for n in args.adapters.split(",")] if args.adapters else None
    adapter_dirs = get_adapter_dirs(names)

    if not adapter_dirs:
        print("No adapters found to archive.", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("  Archiving adapter weights")
    print("=" * 60)

    tarball = create_archive(
        adapter_dirs,
        include_merged=args.include_merged,
        tag=args.tag,
        include_checkpoints=args.include_checkpoints,
    )

    print("\n" + "=" * 60)
    print("  Archive complete.")
    print(f"  Restore: tar xzf {tarball} -C ~/git_home/trainLLM")
    print("=" * 60)


if __name__ == "__main__":
    main()
