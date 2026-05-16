"""Tests for structural_score() and composite_score() in _eval_utils.py."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from _eval_utils import structural_score, composite_score, similarity


def _wrap(obj) -> str:
    """Wrap a dict in a JSON code block the way train_dpo.py formats responses."""
    import json
    return "```json\n" + json.dumps(obj) + "\n```"


class TestStructuralScore:
    def test_exact_match(self):
        call = {"endpoint": "GET /campaigns", "params": {"pageSize": 10}}
        assert structural_score(_wrap(call), _wrap(call)) == 1.0

    def test_same_endpoint_different_params(self):
        expected  = _wrap({"endpoint": "GET /campaigns", "params": {"pageSize": 10, "org": "acme"}})
        generated = _wrap({"endpoint": "GET /campaigns", "params": {"pageSize": 99}})
        score = structural_score(expected, generated)
        # endpoint matches (0.4), partial param score: 1 of 2 keys present (0.5/2) = 0.25 -> 0.6 * 0.25 = 0.15
        assert 0.0 < score < 1.0
        assert score > 0.4  # endpoint match contributes

    def test_wrong_endpoint(self):
        # Wrong endpoint with mismatched params → score near 0
        expected  = _wrap({"endpoint": "GET /campaigns", "params": {"pageSize": 10}})
        generated = _wrap({"endpoint": "POST /pixels",   "params": {"unrelated": 99}})
        score = structural_score(expected, generated)
        # endpoint wrong: 0.4 * 0 = 0; params: pageSize absent → 0; total = 0.0
        assert score == pytest.approx(0.0, abs=0.01)

    def test_wrong_endpoint_empty_params(self):
        # Wrong endpoint but empty params → param score is vacuously 1.0
        expected  = _wrap({"endpoint": "GET /campaigns", "params": {}})
        generated = _wrap({"endpoint": "POST /pixels",   "params": {}})
        score = structural_score(expected, generated)
        # endpoint wrong: 0.4 * 0 = 0; params: nothing expected → 0.6 * 1.0 = 0.6
        assert score == pytest.approx(0.6, abs=0.01)

    def test_missing_json_in_generated(self):
        expected  = _wrap({"endpoint": "GET /campaigns", "params": {"pageSize": 10}})
        generated = "I cannot answer that."
        assert structural_score(expected, generated) == 0.0

    def test_both_sides_no_json(self):
        expected  = "Please clarify your request."
        generated = "I'm sorry, I don't understand."
        score = structural_score(expected, generated)
        # Falls back to text similarity — should be a float between 0 and 1
        assert 0.0 <= score <= 1.0
        assert score == pytest.approx(similarity(expected, generated))

    def test_chain_format_exact_match(self):
        chain = {"steps": [
            {"endpoint": "GET /campaigns", "params": {"pageSize": 10}},
            {"endpoint": "GET /pixels",    "params": {"campaignId": 42}},
        ]}
        assert structural_score(_wrap(chain), _wrap(chain)) == 1.0

    def test_chain_format_partial_match(self):
        expected = _wrap({"steps": [
            {"endpoint": "GET /campaigns", "params": {"pageSize": 10}},
            {"endpoint": "GET /pixels",    "params": {"campaignId": 42}},
        ]})
        generated = _wrap({"steps": [
            {"endpoint": "GET /campaigns", "params": {"pageSize": 10}},
            {"endpoint": "GET /pixels",    "params": {"campaignId": 99}},  # wrong value
        ]})
        score = structural_score(expected, generated)
        assert 0.5 < score < 1.0  # first step perfect, second has wrong value

    def test_chain_shorter_generated(self):
        expected = _wrap({"steps": [
            {"endpoint": "GET /campaigns", "params": {}},
            {"endpoint": "GET /pixels",    "params": {}},
        ]})
        generated = _wrap({"steps": [
            {"endpoint": "GET /campaigns", "params": {}},
            # missing second step
        ]})
        score = structural_score(expected, generated)
        assert 0.0 < score < 1.0  # first step matches, second missing

    def test_params_key_presence_partial(self):
        expected  = _wrap({"endpoint": "GET /campaigns", "params": {"a": 1, "b": 2}})
        generated = _wrap({"endpoint": "GET /campaigns", "params": {"a": 1}})
        score = structural_score(expected, generated)
        # endpoint: 1.0 * 0.4 = 0.4
        # params: key a present+correct (1.0), key b absent (0.0) -> avg = 0.5 -> 0.5 * 0.6 = 0.3
        assert score == pytest.approx(0.7, abs=0.01)

    def test_params_key_wrong_value(self):
        expected  = _wrap({"endpoint": "GET /campaigns", "params": {"pageSize": 10}})
        generated = _wrap({"endpoint": "GET /campaigns", "params": {"pageSize": 99}})
        score = structural_score(expected, generated)
        # endpoint: 0.4, params: key present but wrong value -> 0.5 per key -> 0.6 * 0.5 = 0.3
        assert score == pytest.approx(0.70, abs=0.01)


class TestCompositeScore:
    def test_no_judge(self):
        assert composite_score(1.0, 1.0) == pytest.approx(1.0)
        assert composite_score(0.0, 0.0) == pytest.approx(0.0)

    def test_no_judge_weights(self):
        # 0.3 * sim + 0.7 * structural
        result = composite_score(0.5, 0.8)
        assert result == pytest.approx(0.3 * 0.5 + 0.7 * 0.8)

    def test_with_judge(self):
        result = composite_score(0.5, 0.8, judge=0.6)
        assert result == pytest.approx(0.2 * 0.5 + 0.5 * 0.8 + 0.3 * 0.6)

    def test_judge_none_vs_absent(self):
        assert composite_score(0.5, 0.7) == composite_score(0.5, 0.7, judge=None)
