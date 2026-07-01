#!/usr/bin/env python3
"""
Deterministically clean Lean tactic JSONL splits before training.

This script is intentionally conservative. It normalizes whitespace, drops
empty or duplicate records, and can optionally enforce a strict tactic filter.
It does not reshuffle the data or change the underlying prompt format.

Typical use:
  python3 clean_lean_data.py \
    --input-dir data/lean_stat \
    --output-dir data/lean_stat_clean

The output directory will contain cleaned train/valid/test JSONL files plus a
cleanup_report.json with counts and examples.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

STRICT_FORBIDDEN = ("sorry", "admit")


@dataclass(frozen=True)
class CleanConfig:
    strict: bool = False
    max_state_chars: int | None = None
    max_tactic_chars: int | None = None
    dedupe_across_splits: bool = True


def _norm(text: str) -> str:
    return "\n".join(line.rstrip() for line in str(text).replace("\r\n", "\n").replace("\r", "\n").splitlines()).strip()


def _extract(record: dict) -> tuple[str, str] | None:
    convs = record.get("conversations") or []
    if len(convs) != 2:
        return None
    human, gpt = convs
    if human.get("from") != "human" or gpt.get("from") != "gpt":
        return None
    state_match = re.search(
        r"Given the Lean 4 state:\n(.*?)\nProvide the next tactical step\.",
        str(human.get("value", "")),
        re.S,
    )
    state = _norm(state_match.group(1) if state_match else human.get("value", ""))
    tactic = _norm(gpt.get("value", ""))
    return state, tactic


def _make_record(state: str, tactic: str) -> dict:
    return {
        "conversations": [
            {
                "from": "human",
                "value": (
                    f"Given the Lean 4 state:\n{state}\nProvide the next tactical step."
                ),
            },
            {"from": "gpt", "value": tactic},
        ]
    }


def _reject_reason(state: str, tactic: str, cfg: CleanConfig) -> str | None:
    if not state:
        return "empty_state"
    if not tactic:
        return "empty_tactic"
    if cfg.max_state_chars is not None and len(state) > cfg.max_state_chars:
        return "state_too_long"
    if cfg.max_tactic_chars is not None and len(tactic) > cfg.max_tactic_chars:
        return "tactic_too_long"
    if cfg.strict:
        for token in STRICT_FORBIDDEN:
            if token in tactic:
                return f"forbidden_token:{token}"
    return None


def _load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def clean_split(records: list[dict], cfg: CleanConfig, *, seen: set[tuple[str, str]] | None = None) -> tuple[list[dict], dict]:
    cleaned: list[dict] = []
    reject_counts: dict[str, int] = {}
    reject_examples: dict[str, list] = {}
    seen_pairs = seen if seen is not None else set()

    for raw in records:
        extracted = _extract(raw)
        if extracted is None:
            reject_counts["malformed"] = reject_counts.get("malformed", 0) + 1
            continue
        state, tactic = extracted
        reason = _reject_reason(state, tactic, cfg)
        if reason is not None:
            reject_counts[reason] = reject_counts.get(reason, 0) + 1
            bucket = reject_examples.setdefault(reason, [])
            if len(bucket) < 3:
                bucket.append({"state": state[:200], "tactic": tactic[:200]})
            continue

        key = (state, tactic)
        if cfg.dedupe_across_splits and key in seen_pairs:
            reject_counts["duplicate"] = reject_counts.get("duplicate", 0) + 1
            continue
        seen_pairs.add(key)
        cleaned.append(_make_record(state, tactic))

    report = {
        "input_records": len(records),
        "output_records": len(cleaned),
        "rejected_records": sum(reject_counts.values()),
        "strict": cfg.strict,
        "max_state_chars": cfg.max_state_chars,
        "max_tactic_chars": cfg.max_tactic_chars,
        "dedupe_across_splits": cfg.dedupe_across_splits,
        "reject_counts": reject_counts,
        "reject_examples": reject_examples,
    }
    return cleaned, report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--max-state-chars", type=int, default=None)
    ap.add_argument("--max-tactic-chars", type=int, default=None)
    ap.add_argument("--no-dedupe-across-splits", action="store_true")
    args = ap.parse_args()

    cfg = CleanConfig(
        strict=args.strict,
        max_state_chars=args.max_state_chars,
        max_tactic_chars=args.max_tactic_chars,
        dedupe_across_splits=not args.no_dedupe_across_splits,
    )

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    seen: set[tuple[str, str]] = set()
    overall: dict[str, dict] = {}
    for split in ("train", "valid", "test"):
        in_path = args.input_dir / f"{split}.jsonl"
        records = _load_jsonl(in_path)
        cleaned, report = clean_split(records, cfg, seen=seen if cfg.dedupe_across_splits else None)
        _write_jsonl(out_dir / f"{split}.jsonl", cleaned)
        overall[split] = report

    (out_dir / "cleanup_report.json").write_text(
        json.dumps({"config": cfg.__dict__, "splits": overall}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
