"""Tests for API routes using FastAPI TestClient."""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from web.backend.main import app
    return TestClient(app)


@pytest.fixture
def mock_evals_dir(tmp_path):
    """Create a temp evals dir with sample data."""
    eval_data = {
        "meta": {"timestamp": "2026-05-16_1400", "model": "test-model"},
        "summary": {
            "avg_score": 0.65,
            "avg_composite_score": 0.72,
            "avg_structural_score": 0.78,
            "band_counts": {"EXCELLENT": 3, "GOOD": 5, "PARTIAL": 1, "POOR": 1},
        },
        "results": [
            {
                "id": f"holdout_{i:03d}",
                "question": f"test question {i}",
                "expected": '```json\n{"endpoint": "GET /test"}\n```',
                "generated": '```json\n{"endpoint": "GET /test"}\n```',
                "score": 0.6 + i * 0.03,
                "structural_score": 0.7 + i * 0.02,
                "composite_score": 0.65 + i * 0.025,
                "band": "GOOD" if i < 7 else "EXCELLENT",
                "conventions": ["campaigns"],
            }
            for i in range(10)
        ],
    }
    eval_file = tmp_path / "2026-05-16_1400_test-model.json"
    eval_file.write_text(json.dumps(eval_data))

    judge_data = {
        "meta": {"model": "claude-haiku-4-5-20251001"},
        "summary": {"avg_score": 0.55},
        "results": [{"id": f"holdout_{i:03d}", "judge_score": 0.5 + i * 0.04} for i in range(10)],
    }
    judge_file = tmp_path / "2026-05-16_1400_test-model_llmjudge.json"
    judge_file.write_text(json.dumps(judge_data))

    return tmp_path


class TestHealth:
    def test_health_endpoint(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "adapter" in data


class TestRunsRoutes:
    def test_list_runs(self, client, mock_evals_dir):
        with patch("web.backend.routes.runs.cfg") as mock_cfg:
            mock_cfg.evals_dir = mock_evals_dir
            mock_cfg.adapter_name = "test"
            mock_cfg.lora_dir = mock_evals_dir / "lora"
            resp = client.get("/api/runs")
            assert resp.status_code == 200
            runs = resp.json()
            assert len(runs) == 1
            assert runs[0]["avg_composite_score"] == 0.72

    def test_get_run(self, client, mock_evals_dir):
        with patch("web.backend.routes.runs.cfg") as mock_cfg:
            mock_cfg.evals_dir = mock_evals_dir
            mock_cfg.adapter_name = "test"
            mock_cfg.lora_dir = mock_evals_dir / "lora"
            resp = client.get("/api/runs/2026-05-16_1400_test-model")
            assert resp.status_code == 200
            data = resp.json()
            assert "eval" in data
            assert data["eval"]["summary"]["avg_score"] == 0.65

    def test_get_run_not_found(self, client, mock_evals_dir):
        with patch("web.backend.routes.runs.cfg") as mock_cfg:
            mock_cfg.evals_dir = mock_evals_dir
            resp = client.get("/api/runs/nonexistent")
            assert resp.status_code == 404


class TestEvalsRoutes:
    def test_get_eval(self, client, mock_evals_dir):
        with patch("web.backend.routes.evals.cfg") as mock_cfg:
            mock_cfg.evals_dir = mock_evals_dir
            resp = client.get("/api/evals/2026-05-16_1400_test-model")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["results"]) == 10

    def test_get_eval_records_pagination(self, client, mock_evals_dir):
        with patch("web.backend.routes.evals.cfg") as mock_cfg:
            mock_cfg.evals_dir = mock_evals_dir
            resp = client.get("/api/evals/2026-05-16_1400_test-model/records?limit=5&offset=0")
            assert resp.status_code == 200
            data = resp.json()
            assert data["total"] == 10
            assert len(data["records"]) == 5
            assert data["offset"] == 0

    def test_get_eval_records_filter_band(self, client, mock_evals_dir):
        with patch("web.backend.routes.evals.cfg") as mock_cfg:
            mock_cfg.evals_dir = mock_evals_dir
            resp = client.get("/api/evals/2026-05-16_1400_test-model/records?band=EXCELLENT")
            assert resp.status_code == 200
            data = resp.json()
            for r in data["records"]:
                assert r["band"] == "EXCELLENT"

    def test_band_chart(self, client, mock_evals_dir):
        with patch("web.backend.routes.evals.cfg") as mock_cfg:
            mock_cfg.evals_dir = mock_evals_dir
            resp = client.get("/api/evals/2026-05-16_1400_test-model/charts/bands")
            assert resp.status_code == 200
            fig = resp.json()
            assert "data" in fig
            assert fig["data"][0]["type"] == "pie"

    def test_score_distribution_chart(self, client, mock_evals_dir):
        with patch("web.backend.routes.evals.cfg") as mock_cfg:
            mock_cfg.evals_dir = mock_evals_dir
            resp = client.get("/api/evals/2026-05-16_1400_test-model/charts/scores")
            assert resp.status_code == 200
            fig = resp.json()
            assert fig["data"][0]["type"] == "histogram"


class TestDiagnosticsRoutes:
    def test_timing_empty(self, client):
        with patch("web.backend.routes.diagnostics.cfg") as mock_cfg:
            mock_cfg.events_file = Path("/tmp/nonexistent_events.jsonl")
            resp = client.get("/api/diagnostics/timing")
            assert resp.status_code == 200
            assert resp.json()["steps"] == []

    def test_timing_with_events(self, client, tmp_path):
        events_file = tmp_path / "events.jsonl"
        events = [
            {"event": "step_start", "step": "train", "timestamp": "2026-05-16T10:00:00"},
            {"event": "step_end", "step": "train", "timestamp": "2026-05-16T10:05:00", "duration_sec": 300},
        ]
        events_file.write_text("\n".join(json.dumps(e) for e in events))

        with patch("web.backend.routes.diagnostics.cfg") as mock_cfg:
            mock_cfg.events_file = events_file
            resp = client.get("/api/diagnostics/timing")
            assert resp.status_code == 200
            steps = resp.json()["steps"]
            assert len(steps) == 1
            assert steps[0]["step"] == "train"
            assert steps[0]["duration_sec"] == 300

    def test_timing_chart(self, client, tmp_path):
        events_file = tmp_path / "events.jsonl"
        events = [
            {"event": "step_start", "step": "train", "timestamp": "T1"},
            {"event": "step_end", "step": "train", "timestamp": "T2", "duration_sec": 300},
            {"event": "step_start", "step": "eval", "timestamp": "T2"},
            {"event": "step_end", "step": "eval", "timestamp": "T3", "duration_sec": 60},
        ]
        events_file.write_text("\n".join(json.dumps(e) for e in events))

        with patch("web.backend.routes.diagnostics.cfg") as mock_cfg:
            mock_cfg.events_file = events_file
            resp = client.get("/api/diagnostics/timing/chart")
            assert resp.status_code == 200
            fig = resp.json()
            assert fig["data"][0]["type"] == "bar"


class TestConfigRoutes:
    def test_get_config(self, client):
        resp = client.get("/api/config")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)

    def test_get_training_config(self, client):
        resp = client.get("/api/config/training")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)


class TestCycleRoutes:
    def test_cycle_status_idle(self, client):
        resp = client.get("/api/cycle/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["running"] is False
