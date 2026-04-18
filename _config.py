"""Shared config loader — imported by train.py, eval.py, and cycle.py."""

import os
from pathlib import Path
from types import SimpleNamespace

_HERE = Path(__file__).parent


def load(config_path: Path | None = None) -> SimpleNamespace:
    try:
        import yaml
    except ImportError:
        raise SystemExit(
            "PyYAML not found. Install it: pip install pyyaml\n"
            "Or in the Unsloth env: ~/.unsloth/studio/unsloth_studio/bin/pip install pyyaml"
        )

    path = config_path or (_HERE / "config.yaml")
    with open(path) as f:
        raw = yaml.safe_load(f)

    def exp(s: str) -> Path:
        return Path(os.path.expanduser(str(s)))

    base_dir     = exp(raw["paths"]["base_dir"])
    adapter_name = raw["adapter_name"]

    return SimpleNamespace(
        model         = raw["model"],
        adapter_name  = adapter_name,
        chat_template = raw.get("chat_template", "chatml"),

        base_dir        = base_dir,
        hf_home         = exp(raw["paths"]["hf_home"]),
        unsloth_python  = exp(raw["paths"]["unsloth_python"]),

        data_dir  = base_dir / "data",
        lora_dir  = base_dir / "lora" / adapter_name,
        final_dir = base_dir / "lora" / adapter_name / "final",
        evals_dir = base_dir / "evals",
        logs_dir  = base_dir / "logs",

        train_data = exp(raw["data"]["train"]),
        holdout    = exp(raw["data"]["holdout"]),

        training = SimpleNamespace(**raw["training"]),

        vllm_url                  = f"http://localhost:{raw['vllm']['port']}",
        vllm_port                 = raw["vllm"]["port"],
        vllm_gpu_memory_util      = raw["vllm"]["gpu_memory_utilization"],

        train_silence_timeout = raw["timeouts"]["train_silence"],
        vllm_startup_timeout  = raw["timeouts"]["vllm_startup"],
        vllm_poll_interval    = raw["timeouts"]["vllm_poll_interval"],
        eval_timeout          = raw["timeouts"]["eval_timeout"],
    )
