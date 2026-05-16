"""Tests for prepare_data.py — format conversion, system prompts, splits."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from prepare_data import (
    build_system_prompt,
    format_response,
    to_sharegpt,
    to_holdout,
    stratified_split,
)


class TestBuildSystemPrompt:
    def test_conversational_default(self):
        prompt = build_system_prompt("acme")
        assert "acme" in prompt
        assert "API assistant" in prompt
        assert "JSON" in prompt

    def test_structural_style(self):
        prompt = build_system_prompt("acme", style="structural")
        assert "acme API" in prompt
        assert "<think>" in prompt
        assert len(prompt) < len(build_system_prompt("acme", style="conversational"))

    def test_qoc_style(self):
        prompt = build_system_prompt("acme", style="qoc")
        assert "acme" in prompt
        assert "<think>" in prompt

    def test_org_name_in_prompt(self):
        prompt = build_system_prompt("megacorp")
        assert "megacorp" in prompt

    def test_empty_org_uses_default(self):
        prompt = build_system_prompt("")
        assert "acme" in prompt


class TestFormatResponse:
    def test_simple_api_call(self):
        api_call = {"endpoint": "GET /foo", "params": {"pageSize": 10}}
        result = format_response(api_call)
        assert result.startswith("```json\n")
        assert result.endswith("\n```")
        parsed = json.loads(result.strip("```json\n").strip("\n```"))
        assert parsed == api_call

    def test_with_thinking_trace(self):
        api_call = {"endpoint": "GET /foo", "params": {}}
        thinking = "Entity: foo\nScope: list"
        result = format_response(api_call, thinking=thinking)
        assert result.startswith("<think>\n")
        assert "Entity: foo" in result
        assert "```json\n" in result
        assert result.endswith("\n```")

    def test_without_thinking(self):
        api_call = {"endpoint": "GET /foo", "params": {}}
        result = format_response(api_call, thinking=None)
        assert "<think>" not in result
        assert "```json\n" in result

    def test_chain_format(self):
        api_call = {
            "steps": [
                {"endpoint": "GET /items", "params": {}},
                {"endpoint": "GET /items/{id}", "params": {"id": "{{steps.0.id}}"}},
            ]
        }
        result = format_response(api_call)
        parsed = json.loads(result.strip("```json\n").strip("\n```"))
        assert "steps" in parsed
        assert len(parsed["steps"]) == 2


class TestToSharegpt:
    def test_basic_record(self):
        record = {
            "question": "List measurements",
            "api_call": {"endpoint": "GET /measurements", "params": {"pageSize": 10}},
        }
        result = to_sharegpt(record, system_prompt="You are a test assistant.")
        assert "conversations" in result
        convos = result["conversations"]
        assert len(convos) == 3
        assert convos[0]["from"] == "system"
        assert convos[0]["value"] == "You are a test assistant."
        assert convos[1]["from"] == "human"
        assert convos[1]["value"] == "List measurements"
        assert convos[2]["from"] == "gpt"
        assert "```json" in convos[2]["value"]

    def test_with_thinking(self):
        record = {
            "question": "Get item 5",
            "api_call": {"endpoint": "GET /items/{id}", "params": {"id": 5}},
            "thinking": "Entity: item\nScope: single",
        }
        result = to_sharegpt(record, system_prompt="test")
        gpt_response = result["conversations"][2]["value"]
        assert "<think>" in gpt_response
        assert "Entity: item" in gpt_response

    def test_without_thinking(self):
        record = {
            "question": "List items",
            "api_call": {"endpoint": "GET /items", "params": {}},
        }
        result = to_sharegpt(record, system_prompt="test")
        gpt_response = result["conversations"][2]["value"]
        assert "<think>" not in gpt_response


class TestToHoldout:
    def test_basic_record(self):
        record = {
            "question": "List measurements",
            "api_call": {"endpoint": "GET /measurements", "params": {"pageSize": 10}},
        }
        result = to_holdout(record, endpoint_name="measurements", idx=42, system_prompt="test")
        assert result["id"] == "measurements-0042"
        assert result["label"] == "List measurements"
        assert len(result["messages"]) == 3
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][1]["role"] == "user"
        assert result["messages"][2]["role"] == "assistant"

    def test_conventions_page_size_only(self):
        record = {
            "question": "List",
            "api_call": {"endpoint": "GET /items", "params": {"pageSize": 10}},
        }
        result = to_holdout(record, "items", 0, "test")
        assert "items" in result["conventions_tested"]
        assert "page-size-only" in result["conventions_tested"]

    def test_conventions_no_params(self):
        record = {
            "question": "List",
            "api_call": {"endpoint": "GET /items", "params": {}},
        }
        result = to_holdout(record, "items", 0, "test")
        assert "no-params" in result["conventions_tested"]

    def test_conventions_filtered(self):
        record = {
            "question": "List",
            "api_call": {"endpoint": "GET /items", "params": {"pageSize": 10, "status": "active"}},
        }
        result = to_holdout(record, "items", 0, "test")
        assert "filtered" in result["conventions_tested"]

    def test_conventions_pagination(self):
        record = {
            "question": "Next page",
            "api_call": {"endpoint": "GET /items", "params": {"pageToken": "abc"}},
        }
        result = to_holdout(record, "items", 0, "test")
        assert "pagination" in result["conventions_tested"]


class TestStratifiedSplit:
    def test_basic_split(self):
        records = {
            "endpoint_a": [{"q": i} for i in range(20)],
            "endpoint_b": [{"q": i} for i in range(10)],
        }
        train, holdout = stratified_split(records, holdout_frac=0.1, seed=42)
        assert len(train) + len(holdout) == 30

    def test_at_least_one_holdout_per_endpoint(self):
        records = {
            "tiny": [{"q": 0}, {"q": 1}],
            "also_tiny": [{"q": 0}],
        }
        train, holdout = stratified_split(records, holdout_frac=0.1, seed=42)
        holdout_endpoints = {ep for ep, _ in holdout}
        assert "tiny" in holdout_endpoints
        assert "also_tiny" in holdout_endpoints

    def test_deterministic_with_seed(self):
        records = {"ep": [{"q": i} for i in range(50)]}
        t1, h1 = stratified_split(records, holdout_frac=0.2, seed=42)
        t2, h2 = stratified_split(records, holdout_frac=0.2, seed=42)
        assert t1 == t2
        assert h1 == h2

    def test_different_seeds_different_splits(self):
        records = {"ep": [{"q": i} for i in range(50)]}
        _, h1 = stratified_split(records, holdout_frac=0.2, seed=42)
        _, h2 = stratified_split(records, holdout_frac=0.2, seed=99)
        assert h1 != h2

    def test_holdout_fraction_respected(self):
        records = {"ep": [{"q": i} for i in range(100)]}
        train, holdout = stratified_split(records, holdout_frac=0.2, seed=42)
        assert len(holdout) == 20
        assert len(train) == 80

    def test_no_overlap(self):
        records = {"ep": [{"q": i} for i in range(50)]}
        train, holdout = stratified_split(records, holdout_frac=0.2, seed=42)
        train_qs = {r["q"] for _, r in train}
        holdout_qs = {r["q"] for _, r in holdout}
        assert train_qs.isdisjoint(holdout_qs)
