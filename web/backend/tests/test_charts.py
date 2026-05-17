"""Tests for chart construction — verify Plotly figure specs are valid."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from web.backend.charts import (
    loss_curve, lr_schedule, score_distribution,
    band_donut, timing_bars, composite_trend,
)


class TestLossCurve:
    def test_basic_structure(self):
        fig = loss_curve([1, 2, 3], [1.0, 0.8, 0.6])
        assert "data" in fig
        assert "layout" in fig
        assert len(fig["data"]) == 1
        assert fig["data"][0]["type"] == "scatter"

    def test_with_eval_loss(self):
        fig = loss_curve(
            [1, 2, 3, 4], [1.0, 0.8, 0.6, 0.5],
            eval_loss=[0.9, 0.7],
            eval_steps=[2, 4],
        )
        assert len(fig["data"]) == 2
        assert fig["data"][1]["name"] == "Eval Loss"

    def test_empty_data(self):
        fig = loss_curve([], [])
        assert "data" in fig
        assert len(fig["data"][0]["x"]) == 0

    def test_layout_properties(self):
        fig = loss_curve([1], [1.0])
        layout = fig["layout"]
        assert "template" in layout
        assert layout["xaxis"]["title"]["text"] == "Step"


class TestLRSchedule:
    def test_basic(self):
        fig = lr_schedule([0, 10, 20], [0.0001, 0.0002, 0.0002])
        assert len(fig["data"]) == 1
        assert fig["data"][0]["line"]["color"] == "#10b981"

    def test_has_fill(self):
        fig = lr_schedule([0, 1], [0.0, 0.001])
        assert fig["data"][0]["fill"] == "tozeroy"


class TestScoreDistribution:
    def test_basic(self):
        scores = [0.3, 0.5, 0.7, 0.9, 0.4, 0.6]
        fig = score_distribution(scores)
        assert fig["data"][0]["type"] == "histogram"
        assert list(fig["data"][0]["x"]) == scores

    def test_empty(self):
        fig = score_distribution([])
        assert len(fig["data"][0]["x"]) == 0


class TestBandDonut:
    def test_all_bands(self):
        counts = {"EXCELLENT": 5, "GOOD": 10, "PARTIAL": 3, "POOR": 2}
        fig = band_donut(counts)
        assert fig["data"][0]["type"] == "pie"
        assert fig["data"][0]["hole"] == 0.5
        assert list(fig["data"][0]["values"]) == [5, 10, 3, 2]

    def test_single_band(self):
        fig = band_donut({"GOOD": 100})
        assert list(fig["data"][0]["labels"]) == ["GOOD"]


class TestTimingBars:
    def test_basic(self):
        fig = timing_bars(["train", "eval", "serve"], [300, 60, 15])
        assert fig["data"][0]["type"] == "bar"
        assert fig["data"][0]["orientation"] == "h"
        assert list(fig["data"][0]["y"]) == ["train", "eval", "serve"]

    def test_height_scales_with_items(self):
        fig_small = timing_bars(["a"], [1])
        fig_large = timing_bars(["a", "b", "c", "d", "e", "f"], [1]*6)
        assert fig_large["layout"]["height"] >= fig_small["layout"]["height"]


class TestCompositeTrend:
    def test_basic(self):
        fig = composite_trend(
            ["2026-05-01", "2026-05-02", "2026-05-03"],
            [0.5, 0.6, 0.7],
        )
        assert len(fig["data"]) == 1
        assert fig["data"][0]["name"] == "Composite"

    def test_with_similarity_overlay(self):
        fig = composite_trend(
            ["2026-05-01", "2026-05-02"],
            [0.6, 0.7],
            similarity_scores=[0.4, 0.5],
        )
        assert len(fig["data"]) == 2
        assert fig["data"][1]["name"] == "Similarity"

    def test_y_axis_range(self):
        fig = composite_trend(["d1"], [0.5])
        assert fig["layout"]["yaxis"]["range"] == [0, 1]
