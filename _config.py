"""Shared config loader — imported by train.py, eval.py, and cycle.py."""

import os
from pathlib import Path
from types import SimpleNamespace

_HERE = Path(__file__).parent

_KNOWN_TRAINING_KEYS = {
    "max_seq_length", "lora_rank", "lora_alpha", "lora_dropout",
    "batch_size", "gradient_accumulation_steps", "warmup_steps",
    "max_steps", "learning_rate", "weight_decay", "lr_scheduler",
    "save_steps", "save_total_limit",
}


def load(config_path: Path | None = None) -> SimpleNamespace:
    try:
        import yaml  # type: ignore[import-untyped]
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

    try:
        base_dir     = exp(raw["paths"]["base_dir"])
        adapter_name = raw["adapter_name"]

        unknown_training = set(raw.get("training", {})) - _KNOWN_TRAINING_KEYS
        if unknown_training:
            raise SystemExit(
                f"config.yaml has unknown training keys: {unknown_training}. "
                f"Check for typos. Valid keys: {sorted(_KNOWN_TRAINING_KEYS)}"
            )

        # Inference runtime — "vllm" (default, this machine starts vllm serve)
        # or "external" (an OpenAI-compatible server is started by the user;
        # cycle.py just probes it and skips multi-adapter best-checkpoint).
        runtime = raw.get("runtime", "vllm")
        if runtime not in ("vllm", "external"):
            raise SystemExit(
                f"config.yaml: runtime must be 'vllm' or 'external', got {runtime!r}"
            )
        vllm_block = raw.get("vllm") or {}
        vllm_port = vllm_block.get("port", 8000)
        vllm_gpu_memory_util = vllm_block.get("gpu_memory_utilization", 0.85)

        return SimpleNamespace(
            model         = raw["model"],
            adapter_name  = adapter_name,
            chat_template = raw.get("chat_template", "chatml"),
            runtime       = runtime,

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

            vllm_url                  = f"http://localhost:{vllm_port}",
            vllm_port                 = vllm_port,
            vllm_gpu_memory_util      = vllm_gpu_memory_util,

            train_silence_timeout = raw["timeouts"]["train_silence"],
            vllm_startup_timeout  = raw["timeouts"]["vllm_startup"],
            vllm_poll_interval    = raw["timeouts"]["vllm_poll_interval"],
            eval_timeout          = raw["timeouts"]["eval_timeout"],
        )
    except KeyError as e:
        raise SystemExit(
            f"config.yaml is missing required key: {e}. Verify {path} matches config.example.yaml."
        ) from e
