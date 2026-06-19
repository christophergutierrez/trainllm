#!/usr/bin/env python3
"""Prepare apisynth code-path records for trainLLM SFT.

This is intentionally separate from prepare_data.py so the API path keeps its
existing splitting and prompt behavior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SYSTEM_PROMPT = (
    "You are a code navigation assistant. Given a natural language question "
    "about a repository, reason in <think> tags and then output exactly one "
    "JSON object inside a ```json fenced block. The JSON object identifies the "
    "target code unit with unit, name, file, signature, and class when needed."
)

MAX_NEIGHBOR_NAMES = 6


def _module_path(file_path: str) -> str:
    """Convert a file path to a dot-separated Python module path."""
    p = file_path
    if p.endswith(".py"):
        p = p[:-3]
    parts = p.split("/")
    while parts and parts[0] in ("python", "src"):
        parts = parts[1:]
    return ".".join(parts)


def _enrich_thinking(records: list[dict]) -> None:
    """Append Module/Neighbors lines to thinking fields (in place).

    Builds a file → sibling-name index from all records, then appends
    context lines to each record's thinking so the model learns stronger
    file-location associations.
    """
    file_siblings: dict[str, list[str]] = {}
    for r in records:
        f = r["output"].get("file", "")
        n = r["output"].get("name", "")
        if f and n:
            file_siblings.setdefault(f, []).append(n)

    for r in records:
        thinking = r["thinking"]
        if "\nModule:" in thinking:
            continue
        f = r["output"].get("file", "")
        own_name = r["output"].get("name", "")
        siblings = file_siblings.get(f)
        if not siblings:
            continue
        others = [n for n in siblings if n != own_name]
        if not others:
            continue
        module = _module_path(f)
        sample = others[:MAX_NEIGHBOR_NAMES]
        suffix = f"\nModule: {module}"
        if len(others) > MAX_NEIGHBOR_NAMES:
            suffix += f"\nNeighbors: {', '.join(sample)}, … (+{len(others) - MAX_NEIGHBOR_NAMES} more)"
        else:
            suffix += f"\nNeighbors: {', '.join(sample)}"
        r["thinking"] = thinking + suffix


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("type") != "code":
            raise SystemExit(f"{path}:{line_no}: expected type='code'")
        if not isinstance(record.get("question"), str):
            raise SystemExit(f"{path}:{line_no}: missing string question")
        if not isinstance(record.get("thinking"), str):
            raise SystemExit(f"{path}:{line_no}: missing string thinking")
        if not isinstance(record.get("output"), dict):
            raise SystemExit(f"{path}:{line_no}: missing object output")
        records.append(record)
    return records


def format_response(record: dict) -> str:
    return (
        f"<think>\n{record['thinking']}\n</think>\n"
        "```json\n"
        f"{json.dumps(record['output'], indent=2, ensure_ascii=False)}\n"
        "```"
    )


def to_sharegpt(record: dict, system_prompt: str) -> dict:
    return {
        "conversations": [
            {"from": "system", "value": system_prompt},
            {"from": "human", "value": record["question"]},
            {"from": "gpt", "value": format_response(record)},
        ]
    }


def to_holdout_messages(record: dict, idx: int, system_prompt: str) -> dict:
    output = record["output"]
    conventions = [
        "code-path",
        str(output.get("unit", "unknown-unit")),
        str(output.get("file", "unknown-file")),
    ]
    return {
        "id": f"amesh-code-{idx:04d}",
        "label": record["question"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": record["question"]},
            {"role": "assistant", "content": format_response(record)},
        ],
        "conventions_tested": conventions,
        "source_record": record,
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-in", required=True, type=Path)
    parser.add_argument("--holdout-in", required=True, type=Path)
    parser.add_argument("--train-out", required=True, type=Path)
    parser.add_argument("--holdout-out", required=True, type=Path,
                        help="OpenAI-message holdout prompts for generation.")
    parser.add_argument("--system-prompt", default=SYSTEM_PROMPT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    train_records = load_jsonl(args.train_in.expanduser())
    holdout_records = load_jsonl(args.holdout_in.expanduser())

    all_records = train_records + holdout_records
    _enrich_thinking(all_records)

    train_out = [to_sharegpt(r, args.system_prompt) for r in train_records]
    holdout_out = [
        to_holdout_messages(r, i, args.system_prompt)
        for i, r in enumerate(holdout_records)
    ]

    print(f"train records:   {len(train_records)}")
    print(f"holdout records: {len(holdout_records)}")
    print(f"train out:       {args.train_out.expanduser()}")
    print(f"holdout out:     {args.holdout_out.expanduser()}")

    if args.dry_run:
        return

    write_jsonl(args.train_out.expanduser(), train_out)
    write_jsonl(args.holdout_out.expanduser(), holdout_out)


if __name__ == "__main__":
    main()
