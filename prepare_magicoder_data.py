#!/usr/bin/env python3
"""
Prepare training data from Magicoder-OSS-Instruct-75K and test data from HumanEval+.

Sources:
  Training : ise-uiuc/Magicoder-OSS-Instruct-75K (sample N rows, 95/5 train/holdout)
  Test     : evalplus/humanevalplus (164 problems, held out entirely from training)

Output layout:
    data/magicoder/
        train.jsonl          ShareGPT format, N * 0.95 rows
        holdout.jsonl        ShareGPT format, N * 0.05 rows (used by eval_during_training)
        humaneval_test.jsonl       HumanEval problems, for reference
        humaneval_plus_test.jsonl  HumanEval+ problems, primary benchmark
        manifest.json

Usage:
    python3 prepare_magicoder_data.py
    python3 prepare_magicoder_data.py --sample 25000 --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

from code_verify import clean_completion

DEFAULT_SAMPLE = 25_000
DEFAULT_SEED   = 42
TRAIN_RATIO    = 0.95
DATA_VERSION   = "magicoder-python-clean-v2"

SYSTEM_PROMPT = (
    "You are an expert Python programmer. "
    "Write a complete, correct Python solution for the given problem. "
    "Output only executable Python code. Do not use markdown fences or explanations."
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_jsonl(records: list[dict], path: Path) -> None:
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _is_python_solution(solution: str) -> bool:
    stripped = solution.lstrip()
    if stripped.startswith("```"):
        first = stripped.splitlines()[0].strip().lower()
        return first in {"```python", "```py"}
    return any(token in stripped for token in ("def ", "class ", "import ", "from "))


def make_train_record(problem: str, solution: str) -> dict:
    return {
        "conversations": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": problem.strip()},
            {"role": "assistant", "content": clean_completion(solution)},
        ]
    }


def prepare(output_dir: Path, sample: int, seed: int) -> dict:
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("datasets not found: pip install datasets")

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    # ── Magicoder training corpus ─────────────────────────────────────────────
    print(f"Loading ise-uiuc/Magicoder-OSS-Instruct-75K …")
    magicoder = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train")
    python_rows = [r for r in magicoder if _is_python_solution(r["solution"])]
    rows = list(python_rows)
    if len(rows) < sample:
        print(f"  Requested {sample} rows but only found {len(rows)} Python rows; using all of them")
    rng.shuffle(rows)
    rows = rows[:sample]
    print(f"  Sampled {len(rows)} Python rows / {len(magicoder)} total rows")

    records = [make_train_record(r["problem"], r["solution"]) for r in rows]
    split = int(len(records) * TRAIN_RATIO)
    train_records   = records[:split]
    holdout_records = records[split:]

    train_path   = output_dir / "train.jsonl"
    holdout_path = output_dir / "holdout.jsonl"
    _write_jsonl(train_records,   train_path)
    _write_jsonl(holdout_records, holdout_path)
    print(f"  train:   {len(train_records)} → {train_path}")
    print(f"  holdout: {len(holdout_records)} → {holdout_path}")

    # ── HumanEval test sets ───────────────────────────────────────────────────
    print("Loading openai/openai_humaneval …")
    humaneval = load_dataset("openai/openai_humaneval", split="test")
    he_records = [
        {
            "task_id":     row["task_id"],
            "prompt":      row["prompt"],
            "entry_point": row["entry_point"],
            "test":        row["test"],
            "canonical_solution": row["canonical_solution"],
        }
        for row in humaneval
    ]
    he_path = output_dir / "humaneval_test.jsonl"
    _write_jsonl(he_records, he_path)
    print(f"  humaneval_test: {len(he_records)} problems → {he_path}")

    print("Loading evalplus/humanevalplus …")
    humaneval_plus = load_dataset("evalplus/humanevalplus", split="test")
    he_plus_records = [
        {
            "task_id":     row["task_id"],
            "prompt":      row["prompt"],
            "entry_point": row["entry_point"],
            "test":        row["test"],
            "canonical_solution": row["canonical_solution"],
        }
        for row in humaneval_plus
    ]
    he_plus_path = output_dir / "humaneval_plus_test.jsonl"
    _write_jsonl(he_plus_records, he_plus_path)
    print(f"  humaneval_plus_test: {len(he_plus_records)} problems → {he_plus_path}")

    # ── Manifest ─────────────────────────────────────────────────────────────
    files = [train_path, holdout_path, he_path, he_plus_path]
    manifest = {
        "magicoder_source": "ise-uiuc/Magicoder-OSS-Instruct-75K",
        "data_version": DATA_VERSION,
        "humaneval_source": "openai/openai_humaneval",
        "humaneval_plus_source": "evalplus/humanevalplus",
        "seed": seed,
        "sample": sample,
        "python_records_available": len(python_rows),
        "train_ratio": TRAIN_RATIO,
        "train_records":   len(train_records),
        "holdout_records": len(holdout_records),
        "humaneval_problems": len(he_records),
        "humaneval_plus_problems": len(he_plus_records),
        "files": {
            p.name: {"sha256": _sha256(p), "lines": sum(1 for _ in p.open())}
            for p in files
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  manifest → {manifest_path}")
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=Path("data/magicoder"))
    p.add_argument("--sample", type=int, default=DEFAULT_SAMPLE,
                   help=f"Rows to sample from Magicoder (default {DEFAULT_SAMPLE})")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = p.parse_args()
    prepare(args.output_dir, args.sample, args.seed)
    print("\nDone.")


if __name__ == "__main__":
    main()
