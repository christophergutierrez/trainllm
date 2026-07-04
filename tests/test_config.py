"""Tests for _config.py — config loading, defaults, validation."""

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
import _config


MINIMAL_CONFIG = {
    "model": "test/model",
    "adapter_name": "test-adapter",
    "paths": {
        "base_dir": "/tmp/trainllm-test",
        "hf_home": "/tmp/hf",
        "unsloth_python": "/usr/bin/python3",
    },
    "data": {
        "train": "/tmp/train.jsonl",
        "holdout": "/tmp/holdout.jsonl",
    },
    "training": {
        "max_seq_length": 2048,
    },
    "vllm": {"port": 8000},
    "timeouts": {
        "train_silence": 1800,
        "vllm_startup": 900,
        "vllm_poll_interval": 30,
        "eval_timeout": 3600,
    },
}


def _write_config(tmp_path: Path, overrides: dict | None = None) -> Path:
    cfg = {**MINIMAL_CONFIG}
    if overrides:
        for k, v in overrides.items():
            if isinstance(v, dict) and k in cfg and isinstance(cfg[k], dict):
                cfg[k] = {**cfg[k], **v}
            else:
                cfg[k] = v
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(cfg))
    return path


class TestConfigDefaults:
    def test_loads_minimal_config(self, tmp_path):
        path = _write_config(tmp_path)
        cfg = _config.load(config_path=path)
        assert cfg.model == "test/model"
        assert cfg.adapter_name == "test-adapter"

    def test_training_defaults_applied(self, tmp_path):
        path = _write_config(tmp_path)
        cfg = _config.load(config_path=path)
        assert cfg.training.lora_rank == 16
        assert cfg.training.lora_alpha == 32
        assert cfg.training.lora_dropout == 0.0
        assert cfg.training.batch_size == 2
        assert cfg.training.gradient_accumulation_steps == 4
        assert cfg.training.warmup_steps == 50
        assert cfg.training.max_steps == 2000
        assert cfg.training.learning_rate == 2e-4
        assert cfg.training.weight_decay == 0.01
        assert cfg.training.lr_scheduler == "cosine"
        assert cfg.training.save_steps == 500
        assert cfg.training.save_total_limit is None
        assert cfg.training.load_in_4bit is False

    def test_new_training_defaults(self, tmp_path):
        path = _write_config(tmp_path)
        cfg = _config.load(config_path=path)
        assert cfg.training.neftune_noise_alpha == 5.0
        assert cfg.training.train_on_responses_only is True
        assert cfg.training.lora_init == "gaussian"
        assert cfg.training.use_rslora is True

    def test_derived_paths(self, tmp_path):
        path = _write_config(tmp_path)
        cfg = _config.load(config_path=path)
        assert cfg.lora_dir == Path("/tmp/trainllm-test/lora/test-adapter")
        assert cfg.final_dir == Path("/tmp/trainllm-test/lora/test-adapter/final")
        assert cfg.evals_dir == Path("/tmp/trainllm-test/evals")
        assert cfg.data_dir == Path("/tmp/trainllm-test/data")


class TestConfigOverrides:
    def test_explicit_training_values(self, tmp_path):
        path = _write_config(tmp_path, {"training": {
            "lora_rank": 64,
            "lora_alpha": 128,
            "neftune_noise_alpha": 10,
            "train_on_responses_only": False,
            "lora_init": "loftq",
            "use_rslora": False,
        }})
        cfg = _config.load(config_path=path)
        assert cfg.training.lora_rank == 64
        assert cfg.training.lora_alpha == 128
        assert cfg.training.neftune_noise_alpha == 10.0
        assert cfg.training.train_on_responses_only is False
        assert cfg.training.lora_init == "loftq"
        assert cfg.training.use_rslora is False

    def test_disable_neftune_with_zero(self, tmp_path):
        path = _write_config(tmp_path, {"training": {"neftune_noise_alpha": 0}})
        cfg = _config.load(config_path=path)
        assert cfg.training.neftune_noise_alpha == 0.0

    def test_chat_template_default(self, tmp_path):
        path = _write_config(tmp_path)
        cfg = _config.load(config_path=path)
        assert cfg.chat_template == "chatml"

    def test_chat_template_explicit(self, tmp_path):
        path = _write_config(tmp_path, {"chat_template": "qwen-2.5"})
        cfg = _config.load(config_path=path)
        assert cfg.chat_template == "qwen-2.5"

    def test_vllm_defaults(self, tmp_path):
        path = _write_config(tmp_path)
        cfg = _config.load(config_path=path)
        assert cfg.vllm_port == 8000
        assert cfg.vllm_gpu_memory_util == 0.85
        assert cfg.vllm_model == "test/model"


class TestConfigValidation:
    def test_unknown_training_key_fails(self, tmp_path):
        path = _write_config(tmp_path, {"training": {"bogus_key": 42}})
        with pytest.raises(SystemExit, match="unknown training keys"):
            _config.load(config_path=path)

    def test_missing_model_fails(self, tmp_path):
        cfg = {**MINIMAL_CONFIG}
        del cfg["model"]
        path = tmp_path / "config.yaml"
        path.write_text(yaml.dump(cfg))
        with pytest.raises(SystemExit, match="missing required key"):
            _config.load(config_path=path)

    def test_invalid_runtime_fails(self, tmp_path):
        path = _write_config(tmp_path, {"runtime": "invalid"})
        with pytest.raises(SystemExit, match="must be 'vllm' or 'external'"):
            _config.load(config_path=path)


class TestConfigTypeCasting:
    def test_int_fields_cast(self, tmp_path):
        path = _write_config(tmp_path, {"training": {"lora_rank": "32"}})
        cfg = _config.load(config_path=path)
        assert cfg.training.lora_rank == 32
        assert isinstance(cfg.training.lora_rank, int)

    def test_float_fields_cast(self, tmp_path):
        path = _write_config(tmp_path, {"training": {"learning_rate": "0.001"}})
        cfg = _config.load(config_path=path)
        assert cfg.training.learning_rate == 0.001
        assert isinstance(cfg.training.learning_rate, float)

    def test_bool_fields_cast(self, tmp_path):
        path = _write_config(tmp_path, {"training": {"load_in_4bit": "true"}})
        cfg = _config.load(config_path=path)
        assert cfg.training.load_in_4bit is True

    def test_fp8_default_false(self, tmp_path):
        path = _write_config(tmp_path)
        cfg = _config.load(config_path=path)
        assert cfg.training.load_in_fp8 is False

    def test_fp8_explicit_true(self, tmp_path):
        path = _write_config(tmp_path, {"training": {"load_in_fp8": True}})
        cfg = _config.load(config_path=path)
        assert cfg.training.load_in_fp8 is True

    def test_optimizer_default(self, tmp_path):
        path = _write_config(tmp_path)
        cfg = _config.load(config_path=path)
        assert cfg.training.optimizer == "adamw_torch"

    def test_optimizer_8bit(self, tmp_path):
        path = _write_config(tmp_path, {"training": {"optimizer": "adamw_8bit"}})
        cfg = _config.load(config_path=path)
        assert cfg.training.optimizer == "adamw_8bit"

    def test_extra_keys_preserved(self, tmp_path):
        """Keys not in _TRAINING_DEFAULTS are still accessible if in _KNOWN_TRAINING_KEYS."""
        path = _write_config(tmp_path, {"training": {"neftune_noise_alpha": 7.5}})
        cfg = _config.load(config_path=path)
        assert cfg.training.neftune_noise_alpha == 7.5


class TestRealConfigs:
    """Verify all checked-in config files parse without error."""

    CONFIG_FILES = [
        "config.yaml",
        "config.acme.yaml",
        "config.qwen3-thinking.yaml",
        "config.example.yaml",
        "config.magicoder-0.5b.yaml",
        "config.magicoder-1.5b.yaml",
        "config.magicoder-7b.yaml",
        "config.mbpp-0.5b.yaml",
        "config.mbpp-7b.yaml",
    ]

    @pytest.mark.parametrize("filename", CONFIG_FILES)
    def test_config_loads(self, filename):
        path = Path(__file__).parent.parent / filename
        if not path.exists():
            pytest.skip(f"{filename} not found")
        cfg = _config.load(config_path=path)
        assert cfg.model
        assert cfg.adapter_name
        assert cfg.training.lora_rank > 0
        assert cfg.training.neftune_noise_alpha >= 0
        assert cfg.training.lora_init in ("gaussian", "true", "false", "loftq", "corda", "True", "False")
