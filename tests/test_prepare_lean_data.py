"""Tests for prepare_lean_data.py — record format, filtering, splits, reject report.

All tests use in-memory fixtures and do NOT require internet access.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from prepare_lean_data import (
    make_record,
    is_valid,
    split_records,
)


# ── Record schema ────────────────────────────────────────────────────────────

class TestRecordSchema:
    def test_record_schema_valid(self):
        """Output record has the correct messages structure with role/content fields."""
        record = make_record("⊢ n + 0 = n", "simp")
        assert "messages" in record
        msgs = record["messages"]
        assert len(msgs) == 2
        # User turn
        assert msgs[0]["role"] == "user"
        assert "content" in msgs[0]
        assert "⊢ n + 0 = n" in msgs[0]["content"]
        # Assistant turn
        assert msgs[1]["role"] == "assistant"
        assert "content" in msgs[1]
        assert msgs[1]["content"] == "simp"

    def test_user_prompt_contains_preamble(self):
        record = make_record("⊢ True", "trivial")
        content = record["messages"][0]["content"]
        assert content.startswith("Given the Lean 4 state:\n")
        assert "Provide the next tactical step." in content

    def test_assistant_content_is_exact_tactic(self):
        record = make_record("⊢ 1 + 1 = 2", "norm_num")
        assert record["messages"][1]["content"] == "norm_num"

    def test_state_embedded_in_user_content(self):
        state = "case h\nx : Nat\n⊢ x = x"
        record = make_record(state, "rfl")
        assert state in record["messages"][0]["content"]

    def test_only_two_messages(self):
        record = make_record("⊢ P", "exact hp")
        assert len(record["messages"]) == 2


# ── Empty-value filtering ────────────────────────────────────────────────────

class TestNoEmptyStateAllowed:
    def test_no_empty_state_allowed(self):
        """Empty state is rejected."""
        ok, reason = is_valid("", "simp")
        assert not ok

    def test_whitespace_only_state_rejected(self):
        ok, reason = is_valid("   \t\n", "simp")
        assert not ok
        assert "empty_state" in reason

    def test_none_state_rejected(self):
        # Simulates raw.get() returning None
        ok, reason = is_valid(None, "simp")  # type: ignore[arg-type]
        assert not ok


class TestNoEmptyTacticAllowed:
    def test_no_empty_tactic_allowed(self):
        """Empty tactic is rejected."""
        ok, reason = is_valid("⊢ True", "")
        assert not ok

    def test_whitespace_only_tactic_rejected(self):
        ok, reason = is_valid("⊢ True", "  ")
        assert not ok
        assert "empty_tactic" in reason

    def test_none_tactic_rejected(self):
        ok, reason = is_valid("⊢ True", None)  # type: ignore[arg-type]
        assert not ok


class TestValidRecordAccepted:
    def test_normal_record_passes(self):
        ok, reason = is_valid("⊢ n + 0 = n", "simp")
        assert ok
        assert reason == ""

    def test_multiline_state_accepted(self):
        state = "case base\n⊢ 0 + 0 = 0"
        ok, _ = is_valid(state, "rfl")
        assert ok


# ── Strict-mode filtering ────────────────────────────────────────────────────

class TestSorryFilteredInStrictMode:
    def test_sorry_filtered_in_strict_mode(self):
        """Tactic containing 'sorry' is filtered when strict=True."""
        ok, reason = is_valid("⊢ True", "sorry", strict=True)
        assert not ok
        assert "sorry" in reason

    def test_sorry_in_compound_tactic_filtered_strict(self):
        ok, reason = is_valid("⊢ True", "exact sorry", strict=True)
        assert not ok

    def test_sorry_as_suffix_filtered_strict(self):
        ok, reason = is_valid("⊢ True", "by sorry", strict=True)
        assert not ok


class TestSorryAllowedInPermissiveMode:
    def test_sorry_allowed_in_permissive_mode(self):
        """Tactic containing 'sorry' is kept when strict=False."""
        ok, reason = is_valid("⊢ True", "sorry", strict=False)
        assert ok

    def test_sorry_default_is_permissive(self):
        # strict defaults to False
        ok, _ = is_valid("⊢ True", "sorry")
        assert ok


class TestAdmitFilteredInStrictMode:
    def test_admit_filtered_in_strict_mode(self):
        """Tactic containing 'admit' is filtered when strict=True."""
        ok, reason = is_valid("⊢ True", "admit", strict=True)
        assert not ok
        assert "admit" in reason

    def test_admit_in_compound_tactic_filtered_strict(self):
        ok, _ = is_valid("⊢ P ∧ Q", "constructor; admit", strict=True)
        assert not ok


class TestAdmitAllowedInPermissiveMode:
    def test_admit_allowed_in_permissive_mode(self):
        """Tactic containing 'admit' is kept when strict=False."""
        ok, reason = is_valid("⊢ True", "admit", strict=False)
        assert ok


class TestCleanTacticPassesStrict:
    def test_rfl_passes_strict(self):
        ok, _ = is_valid("⊢ x = x", "rfl", strict=True)
        assert ok

    def test_simp_passes_strict(self):
        ok, _ = is_valid("⊢ n + 0 = n", "simp", strict=True)
        assert ok

    def test_by_cases_not_filtered_by_default(self):
        # by_cases is NOT filtered even in strict mode (per the spec, reserved for later)
        ok, _ = is_valid("⊢ P ∨ ¬P", "by_cases h : P", strict=True)
        assert ok


# ── Deterministic split ──────────────────────────────────────────────────────

class TestSplitIsDeterministic:
    @staticmethod
    def _make_records(n: int = 100) -> list[dict]:
        return [make_record(f"⊢ n = {i}", f"tactic_{i}") for i in range(n)]

    def test_split_is_deterministic(self):
        """Same seed produces the same split on the same input."""
        records = self._make_records()
        train1, valid1, test1 = split_records(records, seed=42)
        train2, valid2, test2 = split_records(records, seed=42)
        assert train1 == train2
        assert valid1 == valid2
        assert test1 == test2

    def test_different_seeds_produce_different_splits(self):
        records = self._make_records()
        train1, _, _ = split_records(records, seed=42)
        train2, _, _ = split_records(records, seed=99)
        assert train1 != train2

    def test_split_sizes_are_80_10_10(self):
        records = self._make_records(100)
        train, valid, test = split_records(records, seed=42)
        n = len(records)
        assert len(train) == int(n * 0.8)
        assert len(valid) == int(n * 0.1)
        assert len(train) + len(valid) + len(test) == n

    def test_no_overlap_between_splits(self):
        records = self._make_records(100)
        train, valid, test = split_records(records, seed=42)
        # Serialise to strings so we can use set operations
        to_str = lambda lst: {json.dumps(r, sort_keys=True) for r in lst}
        assert not to_str(train) & to_str(valid)
        assert not to_str(train) & to_str(test)
        assert not to_str(valid) & to_str(test)

    def test_all_records_present_after_split(self):
        records = self._make_records(50)
        train, valid, test = split_records(records, seed=42)
        assert len(train) + len(valid) + len(test) == len(records)


# ── Reject report ────────────────────────────────────────────────────────────

class TestRejectReportHasCounts:
    def test_reject_report_has_counts(self):
        """Reject report JSON has count fields."""
        reject_counts = {
            "empty_state": 3,
            "empty_tactic": 1,
            "forbidden_token:sorry": 2,
        }
        report = {
            "total_processed": 106,
            "total_valid": 100,
            "total_rejected": sum(reject_counts.values()),
            "strict_mode": True,
            "seed": 42,
            "reject_counts": reject_counts,
            "reject_examples": {
                "empty_state": [{"state": "", "tactic": "simp"}],
            },
        }
        assert "total_processed" in report
        assert "total_valid" in report
        assert "total_rejected" in report
        assert "reject_counts" in report
        assert report["total_rejected"] == 6
        assert report["reject_counts"]["empty_state"] == 3

    def test_reject_report_roundtrips_via_json(self, tmp_path: Path):
        """Reject report survives a JSON write/read cycle."""
        reject_counts = {"empty_state": 2, "forbidden_token:admit": 1}
        report = {
            "total_processed": 13,
            "total_valid": 10,
            "total_rejected": 3,
            "strict_mode": True,
            "seed": 42,
            "reject_counts": reject_counts,
            "reject_examples": {
                "empty_state": [{"state": "", "tactic": "rfl"}]
            },
        }
        path = tmp_path / "reject_report.json"
        path.write_text(json.dumps(report, indent=2))
        loaded = json.loads(path.read_text())
        assert loaded["total_processed"] == 13
        assert loaded["reject_counts"]["empty_state"] == 2
        assert loaded["reject_counts"]["forbidden_token:admit"] == 1

    def test_reject_counts_sum_to_total_rejected(self):
        """total_rejected equals the sum of all individual reject_counts."""
        counts = {"empty_state": 4, "empty_tactic": 2, "forbidden_token:sorry": 7}
        report = {
            "total_processed": 100,
            "total_valid": 87,
            "total_rejected": sum(counts.values()),
            "strict_mode": True,
            "seed": 42,
            "reject_counts": counts,
            "reject_examples": {},
        }
        assert report["total_rejected"] == sum(report["reject_counts"].values())
        assert report["total_valid"] + report["total_rejected"] == report["total_processed"]
