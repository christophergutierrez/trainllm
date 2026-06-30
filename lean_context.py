#!/usr/bin/env python3
"""Deterministic few-shot context retrieval for Lean tactic evaluation."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

STATE_RE = re.compile(r"Given the Lean 4 state:\n(.*?)\nProvide", re.S)
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_'.]*|[⊢∀∃→↔≤≥=+*/^\\-]+|\d+")


@dataclass(frozen=True)
class LeanExample:
    index: int
    state_before: str
    tactic: str
    score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def extract_state_tactic(record: dict, index: int = 0) -> LeanExample:
    """Extract a Lean state/tactic pair from ShareGPT or flat JSON records."""
    conversations = record.get("conversations") or []
    if conversations:
        state_before = ""
        tactic = ""
        for message in conversations:
            role = message.get("from")
            value = message.get("value", "")
            if role == "human":
                match = STATE_RE.search(value)
                state_before = match.group(1).strip() if match else value.strip()
            elif role == "gpt":
                tactic = value.strip()
        return LeanExample(index=index, state_before=state_before, tactic=tactic)

    return LeanExample(
        index=index,
        state_before=str(record.get("state_before", "")).strip(),
        tactic=str(record.get("tactic") or record.get("expected_tactic") or "").strip(),
    )


def load_examples(path: Path, limit: int | None = None) -> list[LeanExample]:
    """Load Lean examples from a JSONL file."""
    examples: list[LeanExample] = []
    with path.open(encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            example = extract_state_tactic(json.loads(line), index=idx)
            if example.state_before and example.tactic:
                examples.append(example)
            if limit is not None and len(examples) >= limit:
                break
    return examples


def _tokens(text: str) -> set[str]:
    return {tok.lower() for tok in TOKEN_RE.findall(text)}


def score_example(query_state: str, candidate: LeanExample) -> float:
    """Score a candidate by lexical overlap with the query state."""
    query = _tokens(query_state)
    cand = _tokens(candidate.state_before)
    if not query or not cand:
        return 0.0
    overlap = len(query & cand)
    return overlap / len(query | cand)


def select_context_examples(
    query_state: str,
    examples: list[LeanExample],
    n_shots: int,
    max_state_chars: int = 1200,
) -> list[LeanExample]:
    """Return the top-N deterministic training examples for a query state."""
    if n_shots <= 0:
        return []

    scored = []
    normalized_query = " ".join(query_state.split())
    for example in examples:
        if len(example.state_before) > max_state_chars:
            continue
        if " ".join(example.state_before.split()) == normalized_query:
            continue
        score = score_example(query_state, example)
        if score <= 0:
            continue
        scored.append(LeanExample(
            index=example.index,
            state_before=example.state_before,
            tactic=example.tactic,
            score=round(score, 6),
        ))

    scored.sort(key=lambda ex: (-ex.score, ex.index))
    return scored[:n_shots]


def format_context_block(examples: list[LeanExample]) -> str:
    """Format retrieved examples for non-chat prompts or debugging."""
    blocks = []
    for i, example in enumerate(examples, 1):
        blocks.append(
            f"Example {i}\n"
            f"State:\n{example.state_before}\n"
            f"Tactic:\n{example.tactic}"
        )
    return "\n\n".join(blocks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--state", required=True, help="Lean state to retrieve context for")
    parser.add_argument("--n-shots", type=int, default=5)
    parser.add_argument("--max-state-chars", type=int, default=1200)
    args = parser.parse_args()

    examples = load_examples(args.train)
    selected = select_context_examples(
        args.state,
        examples,
        n_shots=args.n_shots,
        max_state_chars=args.max_state_chars,
    )
    print(json.dumps([ex.to_dict() for ex in selected], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
