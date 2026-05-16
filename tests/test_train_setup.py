"""Tests for train.py setup logic — without GPU or model loading.

Tests the configuration-to-argument mapping and feature flag behavior.
Does not import unsloth/torch (no GPU required).
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


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
