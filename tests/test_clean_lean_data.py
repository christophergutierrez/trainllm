"""Tests for clean_lean_data.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from clean_lean_data import CleanConfig, clean_split, _make_record


def test_clean_split_normalizes_whitespace_and_keeps_valid_records():
    raw = [{
        "conversations": [
            {"from": "human", "value": "Given the Lean 4 state:\n  n : Nat  \n⊢ n = n\nProvide the next tactical step."},
            {"from": "gpt", "value": "  rfl  "},
        ]
    }]

    cleaned, report = clean_split(raw, CleanConfig())
    assert len(cleaned) == 1
    assert cleaned[0]["conversations"][0]["value"].startswith("Given the Lean 4 state:\nn : Nat")
    assert cleaned[0]["conversations"][1]["value"] == "rfl"
    assert report["output_records"] == 1


def test_clean_split_rejects_malformed_and_empty_records():
    raw = [
        {"conversations": []},
        {"conversations": [{"from": "human", "value": "⊢ True"}]},
        {"conversations": [{"from": "human", "value": ""}, {"from": "gpt", "value": "simp"}]},
    ]

    cleaned, report = clean_split(raw, CleanConfig())
    assert cleaned == []
    assert report["reject_counts"]["malformed"] >= 2
    assert report["reject_counts"]["empty_state"] == 1


def test_clean_split_strict_filters_sorry_and_admit():
    raw = [
        _make_record("⊢ True", "sorry"),
        _make_record("⊢ True", "admit"),
        _make_record("⊢ True", "simp"),
    ]

    cleaned, report = clean_split(raw, CleanConfig(strict=True))
    assert len(cleaned) == 1
    assert cleaned[0]["conversations"][1]["value"] == "simp"
    assert report["reject_counts"]["forbidden_token:sorry"] == 1
    assert report["reject_counts"]["forbidden_token:admit"] == 1


def test_clean_split_dedupes_across_splits_when_shared_seen_set():
    cfg = CleanConfig()
    seen: set[tuple[str, str]] = set()
    first, first_report = clean_split([_make_record("⊢ True", "trivial")], cfg, seen=seen)
    second, second_report = clean_split([_make_record("⊢ True", "trivial")], cfg, seen=seen)

    assert len(first) == 1
    assert second == []
    assert first_report["output_records"] == 1
    assert second_report["reject_counts"]["duplicate"] == 1

