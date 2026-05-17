"""Backend configuration — resolves paths to pipeline data."""

import os
from pathlib import Path

import yaml


class WebConfig:
    def __init__(self):
        self.base_dir = Path(os.environ.get(
            "TRAINLLM_BASE", Path(__file__).parent.parent.parent
        ))
        self.host = os.environ.get("WEB_HOST", "0.0.0.0")
        self.port = int(os.environ.get("WEB_PORT", "8080"))

        pipeline_cfg_path = self.base_dir / "config.yaml"
        self._pipeline_cfg: dict = {}
        if pipeline_cfg_path.exists():
            self._pipeline_cfg = yaml.safe_load(pipeline_cfg_path.read_text()) or {}

        self.evals_dir = self.base_dir / "evals"
        self.lora_dir = self._resolve_path("lora")
        self.data_dir = self.base_dir / "data"
        self.fixtures_dir = Path(__file__).parent.parent / "fixtures"
        self.events_file = Path(os.environ.get(
            "TRAINLLM_EVENTS", "/tmp/trainllm_events.jsonl"
        ))
        self.recommendations_file = self.base_dir / "recommendations.yaml"
        self.agent_inbox = Path(os.environ.get(
            "TRAINLLM_AGENT_INBOX", "/tmp/trainllm_agent_inbox.jsonl"
        ))
        self.agent_outbox = Path(os.environ.get(
            "TRAINLLM_AGENT_OUTBOX", "/tmp/trainllm_agent_outbox.jsonl"
        ))
        self.agent_presence = Path(os.environ.get(
            "TRAINLLM_AGENT_PRESENCE", "/tmp/trainllm_agent_presence.json"
        ))
        self.agent_presence_ttl = int(os.environ.get(
            "TRAINLLM_AGENT_PRESENCE_TTL", "300"
        ))

    def _resolve_path(self, key: str) -> Path:
        return self.base_dir / key

    @property
    def pipeline_config(self) -> dict:
        return self._pipeline_cfg

    @property
    def adapter_name(self) -> str:
        return self._pipeline_cfg.get("adapter_name", "unknown")


cfg = WebConfig()
