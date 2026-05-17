"""Tests for EventEmitterCallback — writes training events to JSONL."""

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from _callbacks import EventEmitterCallback


def _make_args(max_steps=100):
    return SimpleNamespace(max_steps=max_steps, learning_rate=2e-4)


def _make_state(step=0):
    return SimpleNamespace(global_step=step)


class TestEventEmitterCallback:
    def test_train_begin_emits(self, tmp_path):
        f = tmp_path / "events.jsonl"
        cb = EventEmitterCallback(str(f))
        cb.on_train_begin(_make_args(), _make_state(), SimpleNamespace())
        lines = f.read_text().strip().split("\n")
        assert len(lines) == 1
        event = json.loads(lines[0])
        assert event["event"] == "step_start"
        assert event["step"] == "train"
        assert event["max_steps"] == 100
        assert "timestamp" in event

    def test_on_log_emits_loss(self, tmp_path):
        f = tmp_path / "events.jsonl"
        cb = EventEmitterCallback(str(f))
        logs = {"loss": 0.5432, "learning_rate": 0.0002, "epoch": 1.5}
        cb.on_log(_make_args(), _make_state(step=50), SimpleNamespace(), logs=logs)
        event = json.loads(f.read_text().strip())
        assert event["event"] == "loss"
        assert event["step"] == 50
        assert event["value"] == 0.5432
        assert event["lr"] == 0.0002
        assert event["epoch"] == 1.5

    def test_on_log_skips_without_loss(self, tmp_path):
        f = tmp_path / "events.jsonl"
        cb = EventEmitterCallback(str(f))
        cb.on_log(_make_args(), _make_state(), SimpleNamespace(), logs={"grad_norm": 1.0})
        assert not f.exists()

    def test_on_log_skips_none_logs(self, tmp_path):
        f = tmp_path / "events.jsonl"
        cb = EventEmitterCallback(str(f))
        cb.on_log(_make_args(), _make_state(), SimpleNamespace(), logs=None)
        assert not f.exists()

    def test_train_end_emits(self, tmp_path):
        f = tmp_path / "events.jsonl"
        cb = EventEmitterCallback(str(f))
        cb.on_train_end(_make_args(), _make_state(step=100), SimpleNamespace())
        event = json.loads(f.read_text().strip())
        assert event["event"] == "step_end"
        assert event["step"] == "train"

    def test_multiple_events_append(self, tmp_path):
        f = tmp_path / "events.jsonl"
        cb = EventEmitterCallback(str(f))
        cb.on_train_begin(_make_args(), _make_state(), SimpleNamespace())
        cb.on_log(_make_args(), _make_state(10), SimpleNamespace(), logs={"loss": 1.0, "epoch": 0.1})
        cb.on_log(_make_args(), _make_state(20), SimpleNamespace(), logs={"loss": 0.8, "epoch": 0.2})
        cb.on_train_end(_make_args(), _make_state(20), SimpleNamespace())
        lines = f.read_text().strip().split("\n")
        assert len(lines) == 4

    def test_handles_unwritable_path(self, tmp_path):
        cb = EventEmitterCallback("/nonexistent/path/events.jsonl")
        # Should not raise
        cb.on_train_begin(_make_args(), _make_state(), SimpleNamespace())
