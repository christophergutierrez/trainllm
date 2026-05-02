#!/usr/bin/env python3
"""
Prepare, train, and evaluate per-endpoint VideoAmp adapters.

This script keeps synced API data in durable local storage under data/, writes
prepared train/holdout files per endpoint, generates endpoint-specific config
files, and reuses cycle.py for the actual Unsloth + vLLM workflow.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

import _config
from prepare_data import load_endpoint, stratified_split, to_holdout, to_sharegpt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", default="handoff/data/videoamp",
                   help="Ephemeral synced endpoint root")
    p.add_argument("--durable-root", default="data/videoamp",
                   help="Durable ignored storage for endpoint source files")
    p.add_argument("--prepared-root", default="data/videoamp_prepared",
                   help="Prepared train/holdout output root")
    p.add_argument("--config-root", default="data/videoamp_runs",
                   help="Generated config output root")
    p.add_argument("--endpoint", action="append", default=[],
                   help="Endpoint to run. Repeat to select multiple. Default: all")
    p.add_argument("--holdout-frac", type=float, default=0.10,
                   help="Per-endpoint holdout fraction")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducible splits")
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
                   help="Base model for all endpoint adapters")
    p.add_argument("--prepare-only", action="store_true",
                   help="Only sync and prepare data/configs; do not train")
    p.add_argument("--skip-base-eval", action="store_true", default=True,
                   help="Skip base-model eval during cycle runs (default: true)")
    p.add_argument("--keep-best-checkpoint", action="store_true",
                   help="Enable cycle.py best-checkpoint selection")
    return p.parse_args()


def discover_endpoints(source_root: Path) -> list[str]:
    return sorted(
        p.name for p in source_root.iterdir()
        if p.is_dir() and (p / "training.jsonl").exists()
    )


def sync_endpoint(source_root: Path, durable_root: Path, endpoint: str) -> Path:
    src = source_root / endpoint / "training.jsonl"
    dst = durable_root / endpoint / "training.jsonl"
    if not src.exists():
        raise FileNotFoundError(f"Missing source file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def choose_training_params(n_train: int) -> tuple[dict, int]:
    if n_train < 32:
        batch_size = 2
        grad_accum = 1
        target_epochs = 12
        min_steps = 96
    elif n_train < 96:
        batch_size = 4
        grad_accum = 1
        target_epochs = 12
        min_steps = 96
    else:
        batch_size = 8
        grad_accum = 2
        target_epochs = 10
        min_steps = 80

    eff_batch = batch_size * grad_accum
    steps_per_epoch = max(1, -(-n_train // eff_batch))
    max_steps = max(min_steps, target_epochs * steps_per_epoch)
    save_steps = max(25, min(100, max_steps // 4))

    training = {
        "max_seq_length": 512,
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0,
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "warmup_steps": 20,
        "max_steps": max_steps,
        "learning_rate": 2.0e-4,
        "weight_decay": 0.01,
        "lr_scheduler": "cosine",
        "save_steps": save_steps,
        "save_total_limit": None,
    }
    return training, max_steps


def prepare_endpoint_files(
    endpoint_file: Path,
    prepared_root: Path,
    endpoint: str,
    holdout_frac: float,
    seed: int,
) -> tuple[Path, Path, int, int]:
    ep_name, records = load_endpoint(endpoint_file)
    train_items, holdout_items = stratified_split({ep_name: records}, holdout_frac, seed)

    out_dir = prepared_root / endpoint
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "training.jsonl"
    holdout_path = out_dir / "holdout.jsonl"

    with train_path.open("w") as f:
        for _, record in train_items:
            f.write(json.dumps(to_sharegpt(record)) + "\n")

    with holdout_path.open("w") as f:
        for idx, (_, record) in enumerate(holdout_items):
            f.write(json.dumps(to_holdout(record, endpoint, idx)) + "\n")

    return train_path, holdout_path, len(train_items), len(holdout_items)


def write_config(
    repo_root: Path,
    cfg_root: Path,
    endpoint: str,
    model: str,
    train_path: Path,
    holdout_path: Path,
    training: dict,
) -> Path:
    current = _config.load()
    config = {
        "model": model,
        "adapter_name": f"videoamp-api-{endpoint}",
        "chat_template": "qwen-2.5",
        "runtime": "vllm",
        "paths": {
            "base_dir": str(repo_root),
            "hf_home": str(current.hf_home),
            "unsloth_python": str(current.unsloth_python),
        },
        "data": {
            "train": str(train_path),
            "holdout": str(holdout_path),
        },
        "training": training,
        "vllm": {
            "port": current.vllm_port,
            "gpu_memory_utilization": current.vllm_gpu_memory_util,
        },
        "timeouts": {
            "train_silence": current.train_silence_timeout,
            "vllm_startup": current.vllm_startup_timeout,
            "vllm_poll_interval": current.vllm_poll_interval,
            "eval_timeout": current.eval_timeout,
        },
    }

    out_dir = cfg_root / endpoint
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "config.yaml"
    with config_path.open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return config_path


def run_cycle(repo_root: Path, config_path: Path, max_steps: int, skip_base_eval: bool, keep_best_checkpoint: bool) -> int:
    cmd = [sys.executable, "cycle.py", "--steps", str(max_steps)]
    if skip_base_eval:
        cmd.append("--skip-base-eval")
    if not keep_best_checkpoint:
        cmd.append("--no-best-checkpoint")

    env = {**os.environ, "TRAINLLM_CONFIG": str(config_path)}
    return subprocess.run(cmd, cwd=repo_root, env=env).returncode


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    source_root = (repo_root / args.source_root).resolve()
    durable_root = repo_root / args.durable_root
    prepared_root = repo_root / args.prepared_root
    config_root = repo_root / args.config_root

    endpoints = args.endpoint or discover_endpoints(source_root)
    if not endpoints:
        raise SystemExit(f"No endpoints found under {source_root}")

    failures: list[str] = []
    for endpoint in endpoints:
        print(f"\n=== {endpoint} ===", flush=True)
        synced = sync_endpoint(source_root, durable_root, endpoint)
        train_path, holdout_path, n_train, n_holdout = prepare_endpoint_files(
            synced, prepared_root, endpoint, args.holdout_frac, args.seed,
        )
        training, max_steps = choose_training_params(n_train)
        config_path = write_config(
            repo_root, config_root, endpoint, args.model, train_path, holdout_path, training,
        )

        print(
            f"synced={synced} train={n_train} holdout={n_holdout} "
            f"batch={training['batch_size']}x{training['gradient_accumulation_steps']} "
            f"steps={max_steps} config={config_path}",
            flush=True,
        )

        if args.prepare_only:
            continue

        rc = run_cycle(
            repo_root,
            config_path,
            max_steps,
            skip_base_eval=args.skip_base_eval,
            keep_best_checkpoint=args.keep_best_checkpoint,
        )
        if rc != 0:
            failures.append(endpoint)

    if failures:
        print(f"\nFailed endpoints: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
