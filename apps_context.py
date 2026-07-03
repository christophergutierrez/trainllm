#!/usr/bin/env python3
"""Deterministic few-shot context retrieval for APPS coding evaluation."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")


@dataclass(frozen=True)
class AppsExample:
    index: int
    question: str
    solution: str
    difficulty: str
    score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def load_train_examples(path: Path, limit: int | None = None) -> list[AppsExample]:
    """Load training examples from a ShareGPT JSONL file (output of prepare_apps_data.py)."""
    examples: list[AppsExample] = []
    with path.open(encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            convs = rec.get("conversations") or []
            question = solution = difficulty = ""
            for msg in convs:
                role = msg.get("from", "")
                val  = msg.get("value", "")
                if role == "human":
                    question = val.strip()
                elif role == "gpt":
                    solution = val.strip()
            if question and solution:
                examples.append(AppsExample(
                    index=idx,
                    question=question,
                    solution=solution,
                    difficulty=difficulty,
                ))
            if limit is not None and len(examples) >= limit:
                break
    return examples


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in TOKEN_RE.findall(text)}


def score_example(query: str, candidate: AppsExample) -> float:
    """Jaccard similarity between query tokens and candidate question tokens."""
    q = _tokens(query)
    c = _tokens(candidate.question)
    if not q or not c:
        return 0.0
    return len(q & c) / len(q | c)


def select_context_examples(
    query: str,
    examples: list[AppsExample],
    n_shots: int,
    max_question_chars: int = 2000,
    same_difficulty: str | None = None,
) -> list[AppsExample]:
    """Return top-N training examples most similar to the query problem.

    Filters by difficulty tier if same_difficulty is provided.
    """
    if n_shots <= 0:
        return []

    normalized_query = " ".join(query.split())
    scored: list[AppsExample] = []
    for ex in examples:
        if len(ex.question) > max_question_chars:
            continue
        if same_difficulty and ex.difficulty and ex.difficulty != same_difficulty:
            continue
        if " ".join(ex.question.split()) == normalized_query:
            continue
        s = score_example(query, ex)
        if s <= 0:
            continue
        scored.append(AppsExample(
            index=ex.index,
            question=ex.question,
            solution=ex.solution,
            difficulty=ex.difficulty,
            score=round(s, 6),
        ))

    scored.sort(key=lambda ex: (-ex.score, ex.index))
    return scored[:n_shots]


def format_context_block(examples: list[AppsExample]) -> str:
    """Format retrieved examples for injection into the model prompt."""
    blocks = []
    for i, ex in enumerate(examples, 1):
        blocks.append(
            f"Example {i}\n"
            f"Problem:\n{ex.question}\n\n"
            f"Solution:\n{ex.solution}"
        )
    return "\n\n".join(blocks)
