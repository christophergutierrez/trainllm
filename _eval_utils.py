"""Shared evaluation utilities used by eval.py, eval_prompt_baseline.py, and emit_synth_status.py."""

import json
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


def _extract_json(text: str) -> dict | None:
    """Parse the JSON code block into a dict for structural comparison."""
    raw = extract_json_block(text)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None


def _param_score(expected_params: dict, generated_params: dict) -> float:
    """Score parameter matching: 0.5 per key for presence, 0.5 per key for value match."""
    if not expected_params:
        return 1.0
    total = 0.0
    for key, exp_val in expected_params.items():
        if key not in generated_params:
            continue
        total += 0.5
        if generated_params[key] == exp_val:
            total += 0.5
    return total / len(expected_params)


def _score_single_call(expected: dict, generated: dict) -> float:
    exp_endpoint = str(expected.get("endpoint", ""))
    gen_endpoint = str(generated.get("endpoint", ""))
    endpoint_score = 1.0 if exp_endpoint == gen_endpoint else 0.0
    exp_params = expected.get("params") or {}
    gen_params = generated.get("params") or {}
    return 0.4 * endpoint_score + 0.6 * _param_score(exp_params, gen_params)


def structural_score(expected: str, generated: str) -> float:
    """Score API responses by structural JSON comparison (endpoint + params).

    Returns similarity fallback when neither side has JSON (e.g. free-text responses).
    Returns 0.0 when expected has JSON but generated does not.
    """
    exp_json = _extract_json(expected)
    gen_json = _extract_json(generated)

    if exp_json is None:
        return similarity(expected, generated)

    if gen_json is None:
        return 0.0

    if "steps" in exp_json or "steps" in gen_json:
        exp_steps = exp_json.get("steps") or [exp_json]
        gen_steps = gen_json.get("steps") or [gen_json]
        if not exp_steps:
            return 1.0
        scores = [
            _score_single_call(exp_step, gen_steps[i]) if i < len(gen_steps) else 0.0
            for i, exp_step in enumerate(exp_steps)
        ]
        return sum(scores) / len(scores)

    return _score_single_call(exp_json, gen_json)


def composite_score(sim: float, structural: float, judge: float | None = None) -> float:
    """Weighted composite of similarity, structural, and optional LLM-judge scores."""
    if judge is None:
        return sim * 0.3 + structural * 0.7
    return sim * 0.2 + structural * 0.5 + judge * 0.3
