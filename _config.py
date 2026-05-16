"""Shared config loader — imported by train.py, eval.py, and cycle.py."""

import os
from pathlib import Path
from types import SimpleNamespace

_HERE = Path(__file__).parent

_KNOWN_TRAINING_KEYS = {
    "max_seq_length", "lora_rank", "lora_alpha", "lora_dropout",
    "batch_size", "gradient_accumulation_steps", "warmup_steps",
    "max_steps", "learning_rate", "weight_decay", "lr_scheduler",
    "save_steps", "save_total_limit", "load_in_4bit",
}


_TRAINING_DEFAULTS: dict[str, tuple[type, object]] = {
    "max_seq_length":              (int,   2048),
    "lora_rank":                   (int,   16),
    "lora_alpha":                  (int,   32),
    "lora_dropout":                (float, 0),
    "batch_size":                  (int,   2),
    "gradient_accumulation_steps": (int,   4),
    "warmup_steps":                (int,   50),
    "max_steps":                   (int,   2000),
    "learning_rate":               (float, 2e-4),
    "weight_decay":                (float, 0.01),
    "lr_scheduler":                (str,   "cosine"),
    "save_steps":                  (int,   500),
    "save_total_limit":            (None,  None),   # None type = int-or-None
    "load_in_4bit":                (bool,  False),
}


def _parse_training(raw_training: dict) -> SimpleNamespace:
    result = {}
    for key, (typ, default) in _TRAINING_DEFAULTS.items():
        val = raw_training.get(key, default)
        if val is None:
            result[key] = None
        elif typ is None:
            result[key] = int(val) if val is not None else None
        elif typ is bool:
            result[key] = bool(val)
        elif typ is int:
            result[key] = int(val)
        elif typ is float:
            result[key] = float(val)
        else:
            result[key] = str(val)
    for key in raw_training:
        if key not in _TRAINING_DEFAULTS:
            result[key] = raw_training[key]
    return SimpleNamespace(**result)


def load(config_path: Path | None = None) -> SimpleNamespace:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        raise SystemExit(
            "PyYAML not found. Install it: pip install pyyaml\n"
            "Or in the Unsloth env: ~/.unsloth/studio/unsloth_studio/bin/pip install pyyaml"
        )

    env_path = os.environ.get("TRAINLLM_CONFIG")
    if config_path is not None:
        path = config_path
    elif env_path:
        path = Path(env_path).expanduser()
    else:
        path = _HERE / "config.yaml"
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
        vllm_model = vllm_block.get("model", raw["model"])
        vllm_max_model_len = vllm_block.get("max_model_len", raw["training"]["max_seq_length"] * 2)

        merge_raw = raw.get("merge")
        merge_cfg = None
        if merge_raw:
            merge_adapters = merge_raw.get("adapters", "all")
            merge_cfg = SimpleNamespace(
                method=merge_raw.get("method", "dare_ties"),
                density=merge_raw.get("density", 0.5),
                weight=merge_raw.get("weight", 1.0),
                normalize=merge_raw.get("normalize", True),
                output_dir=exp(merge_raw.get("output_dir", str(base_dir / "merged" / adapter_name))),
                adapters=merge_adapters,
            )

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

            training = _parse_training(raw.get("training", {})),

            merge = merge_cfg,

            vllm_url                  = f"http://localhost:{vllm_port}",
            vllm_port                 = vllm_port,
            vllm_gpu_memory_util      = vllm_gpu_memory_util,
            vllm_max_model_len        = vllm_max_model_len,
            vllm_model                = vllm_model,

            train_silence_timeout = raw["timeouts"]["train_silence"],
            vllm_startup_timeout  = raw["timeouts"]["vllm_startup"],
            vllm_poll_interval    = raw["timeouts"]["vllm_poll_interval"],
            eval_timeout          = raw["timeouts"]["eval_timeout"],
        )
    except KeyError as e:
        raise SystemExit(
            f"config.yaml is missing required key: {e}. Verify {path} matches config.example.yaml."
        ) from e
