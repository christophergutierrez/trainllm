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
    return SequenceMatcher(None, strip_fences(a), strip_fences(b)).ratio()


def strip_fences(text: str) -> str:
    text = re.sub(r"^```[\w]*\n?", "", text.strip())
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def extract_json_block(text: str) -> str:
    """Extract the JSON code block, ignoring thinking traces."""
    m = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def json_similarity(a: str, b: str) -> float:
    """Score only the JSON block, ignoring thinking traces."""
    ja = extract_json_block(a)
    jb = extract_json_block(b)
    if not ja and not jb:
        return 1.0
    if not ja or not jb:
        return 0.0
    return SequenceMatcher(None, ja, jb).ratio()


def diagnostics(generated: str, expected: str) -> dict[str, float]:
    gen = strip_fences(generated)
    exp = strip_fences(expected)
    return {
        "length_ratio": round(len(gen) / max(len(exp), 1), 2),
        "json_score": round(json_similarity(generated, expected), 4),
    }
