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


def _build_retrieval_index(records: list[dict]) -> dict[str, dict[str, set[str]]]:
    """Build name → {file → set of classes} index from all records."""
    index: dict[str, dict[str, set[str]]] = {}
    for r in records:
        out = r.get("output", {})
        name = out.get("name", "")
        file = out.get("file", "")
        cls = out.get("class", "")
        if not name or not file:
            continue
        entry = index.setdefault(name, {})
        entry.setdefault(file, set()).add(cls or "_")
    return index


import re

_BACKTICK_RE = re.compile(r"`([^`]+)`")
_CLASS_CONTEXT_RE = re.compile(
    r"(?:on|of|on a|of a|on an|of an)\s+`([^`]+)`", re.IGNORECASE
)


def _extract_identifiers(question: str) -> tuple[str, str | None]:
    backticks = _BACKTICK_RE.findall(question)
    if not backticks:
        return "", None
    class_match = _CLASS_CONTEXT_RE.search(question)
    class_ctx = class_match.group(1) if class_match else None
    if class_ctx and class_ctx in backticks:
        candidates = [b for b in backticks if b != class_ctx]
        target = candidates[0] if candidates else backticks[-1]
    else:
        target = backticks[-1]
    # Handle Class.method syntax (e.g. `VaultClient.__init__`)
    if "." in target and class_ctx is None:
        parts = target.rsplit(".", 1)
        class_ctx = parts[0]
        target = parts[1]
    return target, class_ctx


def _search_index(index, name, class_ctx=None):
    entries = index.get(name)
    if not entries:
        return []
    if class_ctx:
        narrowed = [f for f, classes in entries.items() if class_ctx in classes]
        if narrowed:
            return sorted(narrowed)
    return sorted(entries.keys())


def _retrieval_context(files, name):
    if not files:
        return ""
    if len(files) == 1:
        return f"[Code search: `{name}` found in {files[0]}]"
    listing = ", ".join(files)
    return f"[Code search: `{name}` found in {listing}]"


def _add_retrieval_context(records: list[dict], index: dict) -> None:
    """Prepend retrieval context to each record's question (in place)."""
    for r in records:
        name, class_ctx = _extract_identifiers(r["question"])
        files = _search_index(index, name, class_ctx)
        ctx = _retrieval_context(files, name)
        if ctx:
            r["question"] = f"{ctx}\n\n{r['question']}"


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
    index = _build_retrieval_index(all_records)
    _add_retrieval_context(train_records, index)

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
