"""Tests for generate_candidates.py — rejection sampling scoring and formatting."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from generate_candidates import score_candidate, to_sharegpt, to_dpo_pair, process_record


class TestScoreCandidate:
    def test_identical_response_scores_high(self):
        expected = '```json\n{"endpoint": "GET /resources", "params": {}}\n```'
        score = score_candidate(expected, expected)
        assert score >= 0.95

    def test_empty_response_scores_low(self):
        expected = '```json\n{"endpoint": "GET /resources", "params": {}}\n```'
        score = score_candidate("", expected)
        assert score < 0.2

    def test_partial_match_scores_middle(self):
        expected = '```json\n{"endpoint": "GET /resources", "params": {"page": 1}}\n```'
        generated = '```json\n{"endpoint": "GET /resources", "params": {}}\n```'
        score = score_candidate(generated, expected)
        assert 0.3 < score < 0.95


class TestToSharegpt:
    def test_basic_conversion(self):
        messages = [
            {"role": "system", "content": "You are an API assistant."},
            {"role": "user", "content": "List resources"},
        ]
        result = to_sharegpt(messages, '```json\n{"result": true}\n```')
        assert result["conversations"][0] == {"from": "system", "value": "You are an API assistant."}
        assert result["conversations"][1] == {"from": "human", "value": "List resources"}
        assert result["conversations"][2] == {"from": "gpt", "value": '```json\n{"result": true}\n```'}

    def test_no_system_message(self):
        messages = [{"role": "user", "content": "hello"}]
        result = to_sharegpt(messages, "world")
        assert len(result["conversations"]) == 2
        assert result["conversations"][0]["from"] == "human"
        assert result["conversations"][1]["from"] == "gpt"


class TestToDpoPair:
    def test_dpo_pair_format(self):
        result = {
            "prompt": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "q"},
            ],
            "best": "good answer",
            "worst": "bad answer",
            "best_score": 0.92,
            "worst_score": 0.31,
        }
        pair = to_dpo_pair(result)
        assert pair["chosen"] == "good answer"
        assert pair["rejected"] == "bad answer"
        assert pair["chosen_score"] == 0.92
        assert pair["rejected_score"] == 0.31
        assert pair["conversations"][0] == {"from": "system", "value": "sys"}
        assert pair["conversations"][1] == {"from": "human", "value": "q"}


class TestProcessRecord:
    def test_picks_best_candidate(self):
        record = {
            "id": "test-001",
            "label": "test record",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
            ],
        }

        mock_client = MagicMock()
        choices = []
        for text in ["world", "wrld", "totally wrong", "world!"]:
            choice = MagicMock()
            choice.message.content = text
            choices.append(choice)

        mock_resp = MagicMock()
        mock_resp.choices = choices
        mock_client.chat.completions.create.return_value = mock_resp

        result = process_record(mock_client, "test-model", record, n=4,
                                temperature=0.6, max_tokens=800)
        assert result["id"] == "test-001"
        assert result["best"] == "world"
        assert result["best_score"] >= 0.95
        assert result["n_candidates"] == 4
        assert result["worst_score"] < result["best_score"]

    def test_handles_single_candidate(self):
        record = {
            "id": "single",
            "messages": [
                {"role": "user", "content": "x"},
                {"role": "assistant", "content": "y"},
            ],
        }
        mock_client = MagicMock()
        choice = MagicMock()
        choice.message.content = "y"
        mock_resp = MagicMock()
        mock_resp.choices = [choice]
        mock_client.chat.completions.create.return_value = mock_resp

        result = process_record(mock_client, "m", record, n=1,
                                temperature=0.6, max_tokens=100)
        assert result["n_candidates"] == 1
        assert result["best"] == result["worst"]
