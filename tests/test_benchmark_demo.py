"""Tests for benchmark_demo.py — pure functions, no live vLLM/API server.

Covers the three server-independent pieces the runbook's Step 4 depends on:
record splitting, per-arm summarisation, and the Markdown report writer.
The network-touching arms (query_openai_compat / query_anthropic) are not
exercised here — they need a running server and are covered by the live demo.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from benchmark_demo import split_record, summarize, write_report


class TestSplitRecord:
    def test_drops_assistant_turn(self):
        rec = {"messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]}
        msgs, expected = split_record(rec)
        assert [m["role"] for m in msgs] == ["system", "user"]
        assert expected == "a"

    def test_only_role_and_content_kept(self):
        rec = {"messages": [
            {"role": "user", "content": "q", "name": "extra", "tool_calls": []},
            {"role": "assistant", "content": "a"},
        ]}
        msgs, _ = split_record(rec)
        assert msgs == [{"role": "user", "content": "q"}]

    def test_no_assistant_returns_empty_expected(self):
        rec = {"messages": [{"role": "user", "content": "q"}]}
        msgs, expected = split_record(rec)
        assert msgs == [{"role": "user", "content": "q"}]
        assert expected == ""

    def test_first_assistant_is_expected(self):
        rec = {"messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "first"},
            {"role": "assistant", "content": "second"},
        ]}
        _, expected = split_record(rec)
        assert expected == "first"


class TestSummarize:
    def _row(self, **kw):
        base = {"composite": 0.5, "latency_ms": 100.0,
                "prompt_tokens": 10, "completion_tokens": 20}
        base.update(kw)
        return base

    def test_basic_aggregation(self):
        rows = [
            self._row(composite=0.4, latency_ms=100, completion_tokens=10),
            self._row(composite=0.6, latency_ms=200, completion_tokens=30),
        ]
        s = summarize(rows)
        assert s["n"] == 2
        assert s["errors"] == 0
        assert s["mean_composite"] == 0.5
        assert s["median_latency_ms"] == 150          # median of 100, 200
        assert s["median_completion_tokens"] == 20    # median of 10, 30

    def test_errors_excluded_from_stats(self):
        rows = [
            self._row(composite=0.8),
            {"id": "x", "error": "boom"},
        ]
        s = summarize(rows)
        assert s["n"] == 1
        assert s["errors"] == 1
        assert s["mean_composite"] == 0.8

    def test_all_errors(self):
        rows = [{"id": "1", "error": "a"}, {"id": "2", "error": "b"}]
        s = summarize(rows)
        assert s == {"n": 0, "errors": 2}

    def test_empty(self):
        assert summarize([]) == {"n": 0, "errors": 0}


class TestWriteReport:
    def _summaries(self, frontier=False):
        s = {
            "base":    {"n": 5, "errors": 0, "mean_composite": 0.42,
                        "median_latency_ms": 120, "median_prompt_tokens": 300,
                        "median_completion_tokens": 80},
            "adapter": {"n": 5, "errors": 0, "mean_composite": 0.91,
                        "median_latency_ms": 130, "median_prompt_tokens": 300,
                        "median_completion_tokens": 75},
        }
        if frontier:
            s["frontier"] = {"n": 5, "errors": 0, "mean_composite": 0.30,
                             "median_latency_ms": 900, "median_prompt_tokens": 300,
                             "median_completion_tokens": 200}
        return s

    def test_writes_table_rows(self, tmp_path):
        out = tmp_path / "report.md"
        write_report(out, self._summaries(), n=5, frontier_model=None)
        text = out.read_text()
        assert "# Benchmark: base vs adapter" in text
        assert "vs frontier" not in text
        assert "| base | 5 | 0.42 | 120 | 80 |" in text
        assert "| adapter | 5 | 0.91 | 130 | 75 |" in text

    def test_frontier_section(self, tmp_path):
        out = tmp_path / "report.md"
        write_report(out, self._summaries(frontier=True), n=5,
                     frontier_model="claude-opus-4-8")
        text = out.read_text()
        assert "# Benchmark: base vs adapter vs frontier" in text
        assert "| frontier | 5 | 0.3 | 900 | 200 |" in text
        assert "`claude-opus-4-8`" in text

    def test_zero_n_arm_renders_dashes(self, tmp_path):
        out = tmp_path / "report.md"
        summaries = {"base": {"n": 0, "errors": 3}}
        write_report(out, summaries, n=3, frontier_model=None)
        text = out.read_text()
        assert "| base | 0 (3 errors) | — | — | — |" in text
