#!/usr/bin/env python3
"""
Prepare Lean 4 tactic training data for trainLLM.

Downloads liminho123/lean4-stat-learning-theory-novel from HuggingFace and
produces train/valid/test JSONL splits in trainLLM's ShareGPT format.

Output record format:
    {"conversations": [
        {"from": "human",
         "value": "Given the Lean 4 state:\\n<STATE>\\nProvide the next tactical step."},
        {"from": "gpt", "value": "<TACTIC>"}
    ]}

Usage:
    python3 prepare_lean_data.py [--output-dir OUTPUT_DIR] [--seed SEED] [--strict] [--inspect]
"""

import argparse
import json
import random
import sys
from pathlib import Path

DATASET_ID = "liminho123/lean4-stat-learning-theory-novel"

# Candidate field names tried in order; first match wins.
STATE_FIELD_CANDIDATES = [
    "state", "tactic_state", "goal", "proof_state", "lean_state",
    "before", "before_state",
]
TACTIC_FIELD_CANDIDATES = [
    "tactic", "next_tactic", "tactic_step", "action", "proof_step",
    "after", "command",
]
TRACED_TACTICS_FIELD = "traced_tactics"
TRACED_STATE_FIELD = "state_before"
TRACED_TACTIC_FIELD = "tactic"

# Tokens that are always rejected in strict mode.
STRICT_FORBIDDEN = ["sorry", "admit"]


# ── Pure helper functions (importable by tests without a network call) ──────

def make_record(state: str, tactic: str) -> dict:
    """Return a ShareGPT-format record for a Lean proof state and tactic."""
    return {
        "conversations": [
            {
                "from": "human",
                "value": (
                    f"Given the Lean 4 state:\n{state}\nProvide the next tactical step."
                ),
            },
            {
                "from": "gpt",
                "value": tactic,
            },
        ]
    }


def is_valid(state: str, tactic: str, strict: bool = False) -> tuple[bool, str]:
    """Return (valid, reason) for a state/tactic pair.

    An empty string reason means the record is accepted.  A non-empty reason
    is a short tag describing why the record was rejected.
    """
    if not (state and state.strip()):
        return False, "empty_state"
    if not (tactic and tactic.strip()):
        return False, "empty_tactic"
    if strict:
        for token in STRICT_FORBIDDEN:
            if token in tactic:
                return False, f"forbidden_token:{token}"
    return True, ""


def split_records(
    records: list, seed: int = 42
) -> tuple[list, list, list]:
    """Split records deterministically into (train, valid, test) at 80/10/10."""
    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * 0.8)
    n_valid = int(n * 0.1)
    train = shuffled[:n_train]
    valid = shuffled[n_train : n_train + n_valid]
    test = shuffled[n_train + n_valid :]
    return train, valid, test


def detect_fields(example: dict) -> tuple[str | None, str | None]:
    """Detect state and tactic field names from a dataset example dict."""
    if isinstance(example.get(TRACED_TACTICS_FIELD), list):
        return TRACED_TACTICS_FIELD, TRACED_TACTIC_FIELD
    keys = set(example.keys())
    state_field = next((f for f in STATE_FIELD_CANDIDATES if f in keys), None)
    tactic_field = next((f for f in TACTIC_FIELD_CANDIDATES if f in keys), None)
    return state_field, tactic_field


def extract_pairs(raw: dict, state_field: str, tactic_field: str) -> list[tuple[str, str]]:
    """Extract one or more (state, tactic) pairs from a raw dataset row."""
    if state_field == TRACED_TACTICS_FIELD:
        pairs = []
        for step in raw.get(TRACED_TACTICS_FIELD) or []:
            if not isinstance(step, dict):
                continue
            state = str(step.get(TRACED_STATE_FIELD) or "").strip()
            tactic = str(step.get(TRACED_TACTIC_FIELD) or "").strip()
            pairs.append((state, tactic))
        return pairs

    state = str(raw.get(state_field) or "").strip()
    tactic = str(raw.get(tactic_field) or "").strip()
    return [(state, tactic)]


def write_jsonl(path: Path, records: list[dict]) -> None:
    """Write records as JSONL — one compact JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_reject_report(
    *,
    rows_processed: int,
    pairs_processed: int,
    valid_count: int,
    strict: bool,
    seed: int,
    limit: int | None,
    state_field: str,
    tactic_field: str,
    reject_counts: dict[str, int],
    reject_examples: dict[str, list],
) -> dict:
    """Build the serialized reject-report payload."""
    total_rejected = sum(reject_counts.values())
    return {
        "total_rows_processed": rows_processed,
        "total_pairs_processed": pairs_processed,
        "total_valid": valid_count,
        "total_rejected": total_rejected,
        "strict_mode": strict,
        "seed": seed,
        "limit": limit,
        "field_mapping": {"state": state_field, "tactic": tactic_field},
        "reject_counts": reject_counts,
        "reject_examples": reject_examples,
    }


# ── CLI entry point ─────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Lean 4 tactic training data from HuggingFace."
    )
    parser.add_argument(
        "--output-dir",
        "--out-dir",
        dest="output_dir",
        default="data/lean_stat",
        help="Output directory for JSONL files (default: data/lean_stat)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit valid output records after filtering (useful for smoke tests)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic splits (default: 42)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Filter tactics containing 'sorry' or 'admit'",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print dataset field names and the first record, then exit",
    )
    args = parser.parse_args()

    # ── Load dataset ────────────────────────────────────────────────────────
    print(f"Loading dataset: {DATASET_ID}")
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit(
            "datasets library not found. Install with:\n  pip install datasets"
        )

    dataset = load_dataset(DATASET_ID)
    available_splits = list(dataset.keys())
    print(f"Available splits: {available_splits}")

    # Use the first split for field inspection.
    first_split_name = available_splits[0]
    first_split = dataset[first_split_name]

    if args.inspect:
        example = first_split[0]
        print("\nField names and value previews:")
        for key, val in example.items():
            preview = str(val)[:120].replace("\n", "\\n")
            print(f"  {key!r:30s}: {preview!r}")
        print("\nFirst record (truncated to 200 chars per field):")
        print(json.dumps({k: str(v)[:200] for k, v in example.items()}, indent=2))
        sys.exit(0)

    # ── Auto-detect field names ─────────────────────────────────────────────
    example = first_split[0]
    state_field, tactic_field = detect_fields(example)

    if state_field is None or tactic_field is None:
        print(
            "ERROR: Could not auto-detect field names.\n"
            "Run with --inspect to see available fields.\n"
            f"  Available:      {list(example.keys())}\n"
            f"  State tried:    {STATE_FIELD_CANDIDATES}\n"
            f"  Nested tried:   {TRACED_TACTICS_FIELD}[].{TRACED_STATE_FIELD}/{TRACED_TACTIC_FIELD}\n"
            f"  Tactic tried:   {TACTIC_FIELD_CANDIDATES}"
        )
        sys.exit(1)

    print(f"Field mapping  : state={state_field!r}, tactic={tactic_field!r}")
    print(f"Strict mode    : {args.strict}")
    print(f"Seed           : {args.seed}")
    print(f"Limit          : {args.limit}")

    # ── Collect all records across every split ──────────────────────────────
    all_raw: list = []
    for split_name in available_splits:
        n_before = len(all_raw)
        all_raw.extend(dataset[split_name])
        print(f"  Split {split_name!r}: {len(all_raw) - n_before} records")
    print(f"Total raw records: {len(all_raw)}")

    # ── Filter and convert ──────────────────────────────────────────────────
    valid_records: list[dict] = []
    reject_counts: dict[str, int] = {}
    reject_examples: dict[str, list] = {}

    total_pairs = 0
    stop = False
    for idx, raw in enumerate(all_raw, start=1):
        for state, tactic in extract_pairs(raw, state_field, tactic_field):
            total_pairs += 1

            ok, reason = is_valid(state, tactic, strict=args.strict)
            if not ok:
                reject_counts[reason] = reject_counts.get(reason, 0) + 1
                bucket = reject_examples.setdefault(reason, [])
                if len(bucket) < 3:
                    bucket.append({"state": state[:200], "tactic": tactic[:200]})
                continue

            valid_records.append(make_record(state, tactic))
            if args.limit is not None and len(valid_records) >= args.limit:
                stop = True
                break

        if idx % 1000 == 0:
            print(f"  Processed {idx}/{len(all_raw)} ...")
        if stop:
            break

    n_skipped = sum(reject_counts.values())
    print(
        f"\nProcessed rows : {idx if all_raw else 0}\n"
        f"Processed pairs: {total_pairs}\n"
        f"Valid     : {len(valid_records)}\n"
        f"Skipped   : {n_skipped}"
    )
    for reason, count in sorted(reject_counts.items()):
        print(f"  {reason}: {count}")

    # ── Write reject report ─────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reject_report = build_reject_report(
        rows_processed=idx if all_raw else 0,
        pairs_processed=total_pairs,
        valid_count=len(valid_records),
        strict=args.strict,
        seed=args.seed,
        limit=args.limit,
        state_field=state_field,
        tactic_field=tactic_field,
        reject_counts=reject_counts,
        reject_examples=reject_examples,
    )
    report_path = output_dir / "reject_report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(reject_report, fh, indent=2, ensure_ascii=False)
    print(f"\nReject report  : {report_path}")

    # ── Deterministic 80/10/10 split ────────────────────────────────────────
    train, valid, test = split_records(valid_records, seed=args.seed)
    print(f"Split          : {len(train)} train / {len(valid)} valid / {len(test)} test")

    # ── Write JSONL output ──────────────────────────────────────────────────
    for split_label, records in [("train", train), ("valid", valid), ("test", test)]:
        path = output_dir / f"{split_label}.jsonl"
        write_jsonl(path, records)
        print(f"  {split_label:5s}: {len(records):>6} records → {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
