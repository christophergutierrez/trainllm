"""Tests for _eval_utils.py — scoring, bands, similarity functions."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from _eval_utils import (
    THRESHOLDS,
    band,
    similarity,
    strip_fences,
    extract_json_block,
    json_similarity,
    diagnostics,
)


class TestBand:
    def test_excellent(self):
        assert band(0.8) == "EXCELLENT"
        assert band(0.95) == "EXCELLENT"
        assert band(1.0) == "EXCELLENT"

    def test_good(self):
        assert band(0.6) == "GOOD"
        assert band(0.79) == "GOOD"

    def test_partial(self):
        assert band(0.4) == "PARTIAL"
        assert band(0.59) == "PARTIAL"

    def test_poor(self):
        assert band(0.0) == "POOR"
        assert band(0.39) == "POOR"

    def test_boundary_values(self):
        assert band(THRESHOLDS["excellent"]) == "EXCELLENT"
        assert band(THRESHOLDS["good"]) == "GOOD"
        assert band(THRESHOLDS["partial"]) == "PARTIAL"
        assert band(THRESHOLDS["partial"] - 0.001) == "POOR"


class TestStripFences:
    def test_strips_json_fence(self):
        text = '```json\n{"key": "value"}\n```'
        assert strip_fences(text) == '{"key": "value"}'

    def test_strips_bare_fence(self):
        text = '```\n{"key": "value"}\n```'
        assert strip_fences(text) == '{"key": "value"}'

    def test_no_fences_unchanged(self):
        text = '{"key": "value"}'
        assert strip_fences(text) == '{"key": "value"}'

    def test_strips_surrounding_whitespace(self):
        text = '  ```json\n{"key": "value"}\n```  '
        assert strip_fences(text) == '{"key": "value"}'

    def test_multiline_content(self):
        text = '```json\n{\n  "a": 1,\n  "b": 2\n}\n```'
        assert strip_fences(text) == '{\n  "a": 1,\n  "b": 2\n}'

    def test_language_tag_stripped(self):
        text = '```python\nprint("hello")\n```'
        assert strip_fences(text) == 'print("hello")'


class TestSimilarity:
    def test_identical_strings(self):
        assert similarity("hello", "hello") == 1.0

    def test_completely_different(self):
        assert similarity("abc", "xyz") == 0.0

    def test_partial_match(self):
        score = similarity("hello world", "hello earth")
        assert 0.4 < score < 0.9

    def test_ignores_fences(self):
        a = '```json\n{"endpoint": "GET /foo"}\n```'
        b = '{"endpoint": "GET /foo"}'
        assert similarity(a, b) == 1.0

    def test_symmetric(self):
        a = "foo bar baz"
        b = "foo baz bar"
        assert similarity(a, b) == similarity(b, a)

    def test_empty_strings(self):
        assert similarity("", "") == 1.0

    def test_one_empty(self):
        assert similarity("hello", "") == 0.0


class TestExtractJsonBlock:
    def test_extracts_json_block(self):
        text = '<think>\nsome reasoning\n</think>\n```json\n{"endpoint": "GET /foo"}\n```'
        assert extract_json_block(text) == '{"endpoint": "GET /foo"}'

    def test_no_json_block(self):
        text = "plain text with no code block"
        assert extract_json_block(text) == ""

    def test_bare_fence(self):
        text = '```\n{"endpoint": "GET /foo"}\n```'
        assert extract_json_block(text) == '{"endpoint": "GET /foo"}'

    def test_multiline_json(self):
        text = '```json\n{\n  "endpoint": "GET /foo",\n  "params": {"pageSize": 10}\n}\n```'
        result = extract_json_block(text)
        assert '"endpoint": "GET /foo"' in result
        assert '"pageSize": 10' in result


class TestJsonSimilarity:
    def test_identical_json_blocks(self):
        text = '```json\n{"endpoint": "GET /foo"}\n```'
        assert json_similarity(text, text) == 1.0

    def test_different_json_blocks(self):
        a = '```json\n{"endpoint": "GET /foo"}\n```'
        b = '```json\n{"endpoint": "GET /bar"}\n```'
        score = json_similarity(a, b)
        assert 0.0 < score < 1.0

    def test_both_missing_json(self):
        assert json_similarity("plain text", "other text") == 1.0

    def test_one_missing_json(self):
        a = '```json\n{"endpoint": "GET /foo"}\n```'
        b = "no code block here"
        assert json_similarity(a, b) == 0.0

    def test_ignores_thinking_traces(self):
        a = '<think>\nEntity: foo\n</think>\n```json\n{"endpoint": "GET /foo"}\n```'
        b = '<think>\nEntity: bar\n</think>\n```json\n{"endpoint": "GET /foo"}\n```'
        assert json_similarity(a, b) == 1.0

    def test_different_thinking_same_json(self):
        a = '<think>\nlong reasoning trace here\n</think>\n```json\n{"x": 1}\n```'
        b = '```json\n{"x": 1}\n```'
        assert json_similarity(a, b) == 1.0


class TestDiagnostics:
    def test_equal_length(self):
        d = diagnostics("hello", "world")
        assert d["length_ratio"] == 1.0

    def test_longer_generated(self):
        d = diagnostics("hello world foo", "hello")
        assert d["length_ratio"] > 1.0

    def test_shorter_generated(self):
        d = diagnostics("hi", "hello world")
        assert d["length_ratio"] < 1.0

    def test_empty_expected(self):
        d = diagnostics("something", "")
        assert d["length_ratio"] > 0

    def test_json_score_included(self):
        a = '```json\n{"a": 1}\n```'
        b = '```json\n{"a": 1}\n```'
        d = diagnostics(a, b)
        assert d["json_score"] == 1.0

    def test_json_score_when_no_blocks(self):
        d = diagnostics("plain", "plain")
        assert d["json_score"] == 1.0
