"""Tests for train.py setup logic — without GPU or model loading.

Tests the configuration-to-argument mapping and feature flag behavior.
Does not import unsloth/torch (no GPU required).
"""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

_REPO_ROOT = str(Path(__file__).parent.parent)


class TestLoraInitMapping:
    """Verify the lora_init config value is correctly mapped for Unsloth."""

    def _map_init(self, value: str):
        """Reproduce the mapping logic from train.py without importing torch."""
        lora_init = value
        if lora_init in ("true", "True"):
            lora_init = True
        elif lora_init in ("false", "False"):
            lora_init = False
        return lora_init

    def test_gaussian_passthrough(self):
        assert self._map_init("gaussian") == "gaussian"

    def test_loftq_passthrough(self):
        assert self._map_init("loftq") == "loftq"

    def test_corda_passthrough(self):
        assert self._map_init("corda") == "corda"

    def test_true_string_to_bool(self):
        assert self._map_init("true") is True
        assert self._map_init("True") is True

    def test_false_string_to_bool(self):
        assert self._map_init("false") is False
        assert self._map_init("False") is False


class TestNeftuneConfig:
    """Verify NEFTune alpha is correctly interpreted."""

    def test_positive_alpha_enables(self):
        alpha = 5.0
        result = alpha if alpha and alpha > 0 else None
        assert result == 5.0

    def test_zero_alpha_disables(self):
        alpha = 0.0
        result = alpha if alpha and alpha > 0 else None
        assert result is None

    def test_none_alpha_disables(self):
        alpha = None
        result = alpha if alpha and alpha > 0 else None
        assert result is None


class TestResponseOnlyMarkers:
    """Verify the chat template markers are correct for known templates."""

    CHATML_MARKERS = {
        "instruction_part": "<|im_start|>user\n",
        "response_part": "<|im_start|>assistant\n",
    }

    def test_markers_present_in_chatml_output(self):
        sample_template_output = (
            "<|im_start|>system\nYou are an assistant.<|im_end|>\n"
            "<|im_start|>user\nHello<|im_end|>\n"
            "<|im_start|>assistant\nWorld<|im_end|>\n"
        )
        assert self.CHATML_MARKERS["instruction_part"] in sample_template_output
        assert self.CHATML_MARKERS["response_part"] in sample_template_output

    def test_markers_split_correctly(self):
        template = (
            "<|im_start|>system\nSys<|im_end|>\n"
            "<|im_start|>user\nQ<|im_end|>\n"
            "<|im_start|>assistant\nA<|im_end|>\n"
        )
        parts = template.split(self.CHATML_MARKERS["response_part"])
        assert len(parts) == 2
        assert "A<|im_end|>" in parts[1]


class TestValidationSplit:
    """Verify the 5% eval split logic used in train.py main()."""

    class _MockDataset:
        def __init__(self, n: int):
            self._n = n

        def train_test_split(self, test_size: float, seed: int):
            n_test = max(1, int(self._n * test_size))
            n_train = self._n - n_test
            return {
                "train": self.__class__(n_train),
                "test":  self.__class__(n_test),
            }

        def __len__(self):
            return self._n

    def _apply_split(self, dataset, eval_during_training: bool):
        if eval_during_training:
            split = dataset.train_test_split(test_size=0.05, seed=42)
            train_dataset = split["train"]
            eval_dataset  = split["test"]
        else:
            train_dataset = dataset
            eval_dataset  = None
        return train_dataset, eval_dataset

    def test_split_is_5_percent(self):
        ds = self._MockDataset(200)
        train, eval_ = self._apply_split(ds, eval_during_training=True)
        assert len(train) + len(eval_) == 200
        assert len(eval_) == 10  # 5% of 200

    def test_no_split_when_disabled(self):
        ds = self._MockDataset(100)
        train, eval_ = self._apply_split(ds, eval_during_training=False)
        assert train is ds
        assert eval_ is None

    def test_minimum_one_eval_record(self):
        ds = self._MockDataset(5)
        _, eval_ = self._apply_split(ds, eval_during_training=True)
        assert len(eval_) >= 1

    def test_split_preserves_total_count(self):
        for n in [20, 100, 500]:
            ds = self._MockDataset(n)
            train, eval_ = self._apply_split(ds, eval_during_training=True)
            assert len(train) + len(eval_) == n


class TestCLI:
    """Tests for CLI argument parsing and --dry-run behavior.

    Uses subprocess so we never import torch/unsloth in the test process.
    """

    def test_help_exits_zero(self):
        result = subprocess.run(
            ["python3", "train.py", "--help"],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
        )
        assert result.returncode == 0

    def test_dry_run_exits_zero(self):
        result = subprocess.run(
            [
                "python3", "train.py", "--dry-run",
                "--base-model", "test/model",
                "--output-dir", "/tmp/test",
                "--train-data", "/tmp/fake.jsonl",
                "--max-steps", "5",
            ],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
        )
        assert result.returncode == 0
        assert "test/model" in result.stdout
        assert "/tmp/test" in result.stdout

    def test_all_required_flags_in_help(self):
        result = subprocess.run(
            ["python3", "train.py", "--help"],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
        )
        assert "--base-model" in result.stdout
        assert "--train-data" in result.stdout
        assert "--output-dir" in result.stdout
        assert "--max-steps" in result.stdout
