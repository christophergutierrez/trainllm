#!/usr/bin/env python3
"""
Download and prepare APPS benchmark data for trainLLM.

Fetches codeparrot/apps from HuggingFace and creates reproducible per-tier
holdouts (introductory / interview / competition) plus a training corpus in
ShareGPT format.

Output layout:
    data/apps/
        eval_introductory.jsonl   (--eval-per-tier records, default 200)
        eval_interview.jsonl
        eval_competition.jsonl
        eval_all.jsonl            (concat of all three tiers)
        train.jsonl               (all training problems, one record per solution)
        train_verified.jsonl      (only solutions that pass test cases; with --verify)
        manifest.json             (counts, SHA-256 digests, seed, source)

Usage:
    python3 prepare_apps_data.py
    python3 prepare_apps_data.py --verify          # also write train_verified.jsonl
    python3 prepare_apps_data.py --output-dir data/apps --eval-per-tier 200 --seed 42
    python3 prepare_apps_data.py --validate-manifest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

DATASET_ID = "codeparrot/apps"
TIERS = ("introductory", "interview", "competition")
DEFAULT_EVAL_PER_TIER = 200
DEFAULT_SEED = 42

SYSTEM_PROMPT = (
    "You are a Python programming assistant. "
    "Write a complete, correct Python solution for the given programming problem. "
    "Output only the code — no explanation, no markdown fences."
)


# ── Pure helpers (importable by tests without a network call) ────────────────

def make_train_record(question: str, solution: str, starter_code: str = "") -> dict:
    """Return a ShareGPT-format training record."""
    user_text = question.strip()
    if starter_code and starter_code.strip():
        user_text += f"\n\nStarter code:\n{starter_code.strip()}"
    return {
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": user_text},
            {"from": "gpt", "value": solution.strip()},
        ]
    }


def _verify_worker(args: tuple) -> dict | None:
    """ProcessPoolExecutor worker: return a training record iff the solution passes tests."""
    question, sol, starter, input_output = args
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from apps_verify import safety_scan, execute_solution
    if not safety_scan(sol).safe:
        return None
    result = execute_solution(sol, input_output, timeout=5.0, max_cases=3)
    if result.passed and result.n_total > 0:
        return make_train_record(question, sol, starter)
    return None


def make_eval_record(row: dict) -> dict | None:
    """Convert a raw HuggingFace APPS row into a normalised eval record.

    Returns None if the row lacks usable test cases.
    """
    io_raw = row.get("input_output") or ""
    if not io_raw:
        return None
    try:
        io = json.loads(io_raw) if isinstance(io_raw, str) else io_raw
    except (json.JSONDecodeError, TypeError):
        return None
    # Must have at least one input/output pair
    inputs = io.get("inputs") or []
    outputs = io.get("outputs") or []
    fn_name = io.get("fn_name")
    if not inputs or not outputs:
        return None
    return {
        "problem_id": row.get("problem_id") or row.get("id"),
        "question": (row.get("question") or "").strip(),
        "starter_code": (row.get("starter_code") or "").strip(),
        "input_output": {
            "inputs": inputs,
            "outputs": outputs,
            **({"fn_name": fn_name} if fn_name else {}),
        },
        "difficulty": row.get("difficulty", ""),
        "url": row.get("url", ""),
    }


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_lines(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


# ── Main preparation logic ───────────────────────────────────────────────────

def prepare(output_dir: Path, eval_per_tier: int, seed: int, verify: bool = False) -> dict:
    """Fetch, split, and write APPS data. Returns the manifest dict."""
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("datasets not found. Install: pip install datasets")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {DATASET_ID} …", flush=True)
    ds = load_dataset(DATASET_ID, trust_remote_code=True)

    rng = random.Random(seed)

    # ── Eval splits ──────────────────────────────────────────────────────────
    test_rows = list(ds["test"])

    tier_rows: dict[str, list[dict]] = {t: [] for t in TIERS}
    for row in test_rows:
        diff = (row.get("difficulty") or "").lower()
        if diff in tier_rows:
            rec = make_eval_record(row)
            if rec:
                tier_rows[diff].append(rec)

    all_eval: list[dict] = []
    eval_counts: dict[str, int] = {}
    for tier in TIERS:
        pool = tier_rows[tier]
        rng.shuffle(pool)
        holdout = pool[:eval_per_tier]
        out_path = output_dir / f"eval_{tier}.jsonl"
        _write_jsonl(holdout, out_path)
        all_eval.extend(holdout)
        eval_counts[tier] = len(holdout)
        print(f"  {tier}: {len(pool)} usable → {len(holdout)} holdout → {out_path}")

    all_path = output_dir / "eval_all.jsonl"
    _write_jsonl(all_eval, all_path)

    # ── Training corpus ──────────────────────────────────────────────────────
    train_records: list[dict] = []
    for row in ds["train"]:
        question = (row.get("question") or "").strip()
        starter = (row.get("starter_code") or "").strip()
        sols_raw = row.get("solutions") or "[]"
        try:
            solutions = json.loads(sols_raw) if isinstance(sols_raw, str) else sols_raw
        except (json.JSONDecodeError, TypeError):
            solutions = []
        for sol in solutions:
            if sol and isinstance(sol, str) and sol.strip():
                train_records.append(make_train_record(question, sol, starter))

    rng.shuffle(train_records)
    train_path = output_dir / "train.jsonl"
    _write_jsonl(train_records, train_path)
    print(f"  train: {len(train_records)} problem×solution pairs → {train_path}")

    # ── Verified training corpus (optional) ──────────────────────────────────
    verified_path = None
    n_verified = 0
    if verify:
        # Build (question, sol, starter, input_output) tuples for all candidates
        # that have test cases in the training split.
        candidates: list[tuple] = []
        skipped_no_tests = 0
        for row in ds["train"]:
            question = (row.get("question") or "").strip()
            starter  = (row.get("starter_code") or "").strip()
            io_raw   = row.get("input_output") or ""
            try:
                io = json.loads(io_raw) if isinstance(io_raw, str) else (io_raw or {})
            except (json.JSONDecodeError, TypeError):
                io = {}
            if not io or not io.get("inputs"):
                skipped_no_tests += 1
                continue
            sols_raw = row.get("solutions") or "[]"
            try:
                solutions = json.loads(sols_raw) if isinstance(sols_raw, str) else sols_raw
            except (json.JSONDecodeError, TypeError):
                solutions = []
            for sol in solutions:
                if sol and isinstance(sol, str) and sol.strip():
                    candidates.append((question, sol, starter, io))

        n_workers = min(multiprocessing.cpu_count(), 16)
        print(f"  Verifying {len(candidates)} solutions ({skipped_no_tests} problems skipped "
              f"— no test cases) using {n_workers} workers...")

        verified_records: list[dict] = []
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_verify_worker, c): i for i, c in enumerate(candidates)}
            for n, future in enumerate(as_completed(futures), 1):
                result = future.result()
                if result is not None:
                    verified_records.append(result)
                if n % 2000 == 0 or n == len(candidates):
                    pct = 100 * len(verified_records) / n
                    print(f"  {n}/{len(candidates)} checked, {len(verified_records)} passing ({pct:.1f}%)",
                          flush=True)

        rng.shuffle(verified_records)
        verified_path = output_dir / "train_verified.jsonl"
        _write_jsonl(verified_records, verified_path)
        n_verified = len(verified_records)
        pass_rate = 100 * n_verified / max(1, len(candidates))
        print(f"  train_verified: {n_verified} records ({pass_rate:.1f}% pass rate) → {verified_path}")

    # ── Manifest ─────────────────────────────────────────────────────────────
    files = [
        output_dir / f"eval_{t}.jsonl" for t in TIERS
    ] + [all_path, train_path] + ([verified_path] if verified_path else [])

    manifest = {
        "source": DATASET_ID,
        "seed": seed,
        "eval_per_tier": eval_per_tier,
        "eval_counts": eval_counts,
        "train_records": len(train_records),
        "train_verified_records": n_verified if verify else None,
        "files": {
            p.name: {
                "path": str(p),
                "lines": _count_lines(p),
                "sha256": _sha256_of_file(p),
            }
            for p in files
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  manifest → {manifest_path}")
    return manifest


def validate_manifest(output_dir: Path) -> bool:
    """Re-hash all files and compare against the stored manifest."""
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        print("ERROR: manifest.json not found. Run without --validate-manifest first.")
        return False

    manifest = json.loads(manifest_path.read_text())
    ok = True
    for name, meta in manifest["files"].items():
        path = Path(meta["path"])
        if not path.exists():
            print(f"MISSING: {path}")
            ok = False
            continue
        actual_hash = _sha256_of_file(path)
        actual_lines = _count_lines(path)
        hash_ok = actual_hash == meta["sha256"]
        line_ok = actual_lines == meta["lines"]
        status = "OK" if (hash_ok and line_ok) else "MISMATCH"
        print(f"  [{status}] {name}: lines={actual_lines}/{meta['lines']} sha256={hash_ok}")
        if not (hash_ok and line_ok):
            ok = False
    return ok


def _write_jsonl(records: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--output-dir", type=Path, default=Path("data/apps"))
    p.add_argument("--eval-per-tier", type=int, default=DEFAULT_EVAL_PER_TIER,
                   help=f"Holdout size per difficulty tier (default: {DEFAULT_EVAL_PER_TIER})")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--validate-manifest", action="store_true",
                   help="Re-hash files and compare against stored manifest, then exit")
    p.add_argument("--verify", action="store_true",
                   help="Also generate train_verified.jsonl by running each solution "
                        "against test cases in parallel (slow, ~30-60 min)")
    args = p.parse_args()

    if args.validate_manifest:
        ok = validate_manifest(args.output_dir)
        sys.exit(0 if ok else 1)

    manifest = prepare(args.output_dir, args.eval_per_tier, args.seed, verify=args.verify)
    print("\nDone.")
    print(json.dumps(manifest["eval_counts"], indent=2))


if __name__ == "__main__":
    main()
