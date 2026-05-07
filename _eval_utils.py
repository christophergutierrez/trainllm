"""Shared evaluation utilities used by eval.py, eval_prompt_baseline.py, and emit_synth_status.py."""

import re
from difflib import SequenceMatcher

THRESHOLDS = {"excellent": 0.8, "good": 0.6, "partial": 0.4}


def band(score: float) -> str:
    if score >= THRESHOLDS["excellent"]:
        return "EXCELLENT"
    if score >= THRESHOLDS["good"]:
        return "GOOD"
    if score >= THRESHOLDS["partial"]:
        return "PARTIAL"
    return "POOR"


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.strip(), b.strip()).ratio()


def strip_fences(text: str) -> str:
    text = re.sub(r"^```[\w]*\n?", "", text.strip())
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def diagnostics(generated: str, expected: str) -> dict[str, float]:
    gen = strip_fences(generated)
    exp = strip_fences(expected)
    return {"length_ratio": round(len(gen) / max(len(exp), 1), 2)}
