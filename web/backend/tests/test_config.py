"""Tests for backend configuration."""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


class TestWebConfig:
    def test_default_host_and_port(self):
        with patch.dict(os.environ, {}, clear=False):
            # Re-import to get fresh config
            from web.backend.config import WebConfig
            c = WebConfig()
            assert c.host == "0.0.0.0"
            assert c.port == 8080

    def test_env_override_port(self):
        with patch.dict(os.environ, {"WEB_PORT": "9999"}):
            from web.backend.config import WebConfig
            c = WebConfig()
            assert c.port == 9999

    def test_base_dir_resolves(self):
        from web.backend.config import WebConfig
        c = WebConfig()
        assert c.base_dir.exists()
        assert (c.base_dir / "train.py").exists()

    def test_evals_dir_path(self):
        from web.backend.config import WebConfig
        c = WebConfig()
        assert c.evals_dir == c.base_dir / "evals"

    def test_pipeline_config_loaded(self):
        from web.backend.config import WebConfig
        c = WebConfig()
        assert isinstance(c.pipeline_config, dict)
        if (c.base_dir / "config.yaml").exists():
            assert "model" in c.pipeline_config or "training" in c.pipeline_config

    def test_adapter_name(self):
        from web.backend.config import WebConfig
        c = WebConfig()
        # Should return something even if config is empty
        assert isinstance(c.adapter_name, str)

    def test_agent_provider_default(self):
        from web.backend.config import WebConfig
        with patch.dict(os.environ, {}, clear=False):
            c = WebConfig()
            assert c.agent_provider == "claude"

    def test_agent_provider_override(self):
        from web.backend.config import WebConfig
        with patch.dict(os.environ, {"AGENT_PROVIDER": "gemini"}):
            c = WebConfig()
            assert c.agent_provider == "gemini"
