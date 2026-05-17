"""Tests for distill.py — teacher model distillation formatting and extraction."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from distill import _extract_prompt_parts, _to_sharegpt


class TestExtractPromptParts:
    def test_sharegpt_format(self):
        record = {
            "conversations": [
                {"from": "system", "value": "You are an API assistant."},
                {"from": "human", "value": "List all users"},
                {"from": "gpt", "value": "```json\n{}\n```"},
            ]
        }
        system, user = _extract_prompt_parts(record)
        assert system == "You are an API assistant."
        assert user == "List all users"

    def test_messages_format(self):
        record = {
            "messages": [
                {"role": "system", "content": "sys prompt"},
                {"role": "user", "content": "user query"},
                {"role": "assistant", "content": "response"},
            ]
        }
        system, user = _extract_prompt_parts(record)
        assert system == "sys prompt"
        assert user == "user query"

    def test_no_system_message(self):
        record = {
            "conversations": [
                {"from": "human", "value": "hello"},
                {"from": "gpt", "value": "hi"},
            ]
        }
        system, user = _extract_prompt_parts(record)
        assert system == ""
        assert user == "hello"

    def test_empty_record(self):
        system, user = _extract_prompt_parts({})
        assert system == ""
        assert user == ""

    def test_messages_no_system(self):
        record = {
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ]
        }
        system, user = _extract_prompt_parts(record)
        assert system == ""
        assert user == "question"


class TestToSharegpt:
    def test_with_system(self):
        result = _to_sharegpt("sys", "user msg", "response")
        assert result == {
            "conversations": [
                {"from": "system", "value": "sys"},
                {"from": "human", "value": "user msg"},
                {"from": "gpt", "value": "response"},
            ]
        }

    def test_without_system(self):
        result = _to_sharegpt("", "user msg", "response")
        assert result == {
            "conversations": [
                {"from": "human", "value": "user msg"},
                {"from": "gpt", "value": "response"},
            ]
        }

    def test_preserves_content(self):
        code = '```json\n{"endpoint": "GET /api/v1/users", "params": {"page": 1}}\n```'
        result = _to_sharegpt("system", "List users page 1", code)
        assert result["conversations"][2]["value"] == code
