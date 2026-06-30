"""Tests for lean_verify.py — all tests that don't need Lean pass on any machine."""

import json
import shutil
import textwrap
from pathlib import Path

import pytest

from lean_verify import (
    VerifyResult,
    _find_lean,
    _make_lean_snippet,
    _parse_state,
    safety_check,
    verify_file,
    verify_tactic,
)

LEAN_AVAILABLE = _find_lean() is not None

# ---------------------------------------------------------------------------
# Safety checks (pure Python — always run)
# ---------------------------------------------------------------------------

class TestSafetyCheck:
    def test_clean_tactic_passes(self):
        assert safety_check("simp") is True

    def test_sorry_rejected(self):
        assert safety_check("sorry") is False

    def test_sorry_in_phrase_rejected(self):
        assert safety_check("by sorry") is False

    def test_admit_rejected(self):
        assert safety_check("admit") is False

    def test_word_boundary_sorry_in_identifier(self):
        # "notsorry" is not a forbidden token
        assert safety_check("notsorry") is True

    def test_multiline_tactic_with_sorry(self):
        assert safety_check("intro h\nsorry") is False

    def test_rfl_passes(self):
        assert safety_check("rfl") is True

    def test_exact_h_symm_passes(self):
        assert safety_check("exact h.symm") is True


# ---------------------------------------------------------------------------
# State parser
# ---------------------------------------------------------------------------

class TestParseState:
    def test_simple_nat_goal(self):
        hyps, goal = _parse_state("n : Nat\n⊢ n + 0 = n")
        assert hyps == ["n : Nat"]
        assert goal == "n + 0 = n"

    def test_no_hypotheses(self):
        hyps, goal = _parse_state("⊢ True")
        assert hyps == []
        assert goal == "True"

    def test_multiple_hyps(self):
        hyps, goal = _parse_state("a b : Nat\nh : a = b\n⊢ b = a")
        assert len(hyps) == 2  # "a b : Nat" and "h : a = b" are two lines
        assert goal == "b = a"

    def test_missing_goal_returns_none(self):
        hyps, goal = _parse_state("n : Nat")
        assert goal is None

    def test_instance_dummy_becomes_typeclass_binder(self):
        hyps, goal = _parse_state("inst✝ : Inhabited Nat\n⊢ True")
        assert hyps == ["[Inhabited Nat]"]
        assert goal == "True"


# ---------------------------------------------------------------------------
# Snippet builder
# ---------------------------------------------------------------------------

class TestMakeLeanSnippet:
    def test_simple_goal(self):
        s = _make_lean_snippet("n : Nat\n⊢ n + 0 = n", "simp")
        assert s is not None
        assert "example" in s
        assert "simp" in s
        assert "(n : Nat)" in s

    def test_no_hyps(self):
        s = _make_lean_snippet("⊢ True", "trivial")
        assert s is not None
        assert "True" in s
        assert "trivial" in s

    def test_instance_state_becomes_typeclass_binder(self):
        s = _make_lean_snippet("inst✝ : Inhabited Nat\n⊢ True", "trivial")
        assert s is not None
        assert "[Inhabited Nat]" in s

    def test_universe_declaration_for_type_state(self):
        s = _make_lean_snippet("A : Type u_1\n⊢ True", "trivial")
        assert s is not None
        assert "universe u_1" in s

    def test_missing_goal_returns_none(self):
        s = _make_lean_snippet("n : Nat", "simp")
        assert s is None


# ---------------------------------------------------------------------------
# verify_tactic (requires Lean; skipped if lean not in PATH)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not LEAN_AVAILABLE, reason="lean not in PATH")
class TestVerifyTacticWithLean:
    def test_valid_simp_passes(self):
        result = verify_tactic("n : Nat\n⊢ n + 0 = n", "simp")
        assert result.passed is True
        assert result.lean_ok is True

    def test_invalid_tactic_fails(self):
        result = verify_tactic("n : Nat\n⊢ n + 0 = n", "rfl")
        assert result.lean_ok is False

    def test_sorry_blocked_before_lean(self):
        result = verify_tactic("⊢ True", "sorry")
        assert result.passed is False
        assert result.safety_ok is False
        assert result.lean_ok is None  # lean never called

    def test_trivial_for_true(self):
        result = verify_tactic("⊢ True", "trivial")
        assert result.passed is True

    def test_rfl_for_eq(self):
        result = verify_tactic("⊢ 1 + 1 = 2", "rfl")
        assert result.passed is True


# ---------------------------------------------------------------------------
# verify_tactic (lean NOT available)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(LEAN_AVAILABLE, reason="lean IS in PATH — run the lean tests instead")
class TestVerifyTacticNoLean:
    def test_sorry_still_blocked(self):
        result = verify_tactic("⊢ True", "sorry")
        assert result.passed is False
        assert result.safety_ok is False

    def test_safe_tactic_lean_none(self):
        result = verify_tactic("⊢ True", "trivial")
        assert result.safety_ok is True
        assert result.lean_ok is None
        assert "lean executable not found" in result.stderr or "cannot be reconstructed" in result.stderr


# ---------------------------------------------------------------------------
# verify_file
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not LEAN_AVAILABLE, reason="lean not in PATH")
class TestVerifyFile:
    def test_fixture_file_compiles(self):
        fixture = Path(__file__).parent.parent / "eval" / "lean_harness" / "Fixture.lean"
        result = verify_file(fixture)
        assert result.passed is True
        assert result.lean_ok is True

    def test_missing_file_raises(self, tmp_path):
        result = verify_file(tmp_path / "nonexistent.lean")
        assert result.passed is False
