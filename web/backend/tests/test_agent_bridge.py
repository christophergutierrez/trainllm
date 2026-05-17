"""Tests for the agent bridge abstraction."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from web.backend.agent_bridge import (
    AgentBridge, ClaudeBridge, GeminiBridge, CodexBridge, create_bridge,
)


class TestBridgeInterface:
    """Verify all bridge implementations satisfy the interface."""

    @pytest.mark.parametrize("bridge_cls", [ClaudeBridge, GeminiBridge, CodexBridge])
    def test_has_required_methods(self, bridge_cls):
        bridge = bridge_cls()
        assert hasattr(bridge, "spawn")
        assert hasattr(bridge, "send")
        assert hasattr(bridge, "receive")
        assert hasattr(bridge, "shutdown")
        assert hasattr(bridge, "is_alive")

    @pytest.mark.parametrize("bridge_cls", [ClaudeBridge, GeminiBridge, CodexBridge])
    def test_starts_not_alive(self, bridge_cls):
        bridge = bridge_cls()
        assert bridge.is_alive is False

    def test_is_abstract(self):
        with pytest.raises(TypeError):
            AgentBridge()


class TestCreateBridge:
    def test_claude(self):
        from unittest.mock import patch
        with patch("web.backend.agent_bridge.cfg") as mock_cfg:
            mock_cfg.agent_provider = "claude"
            bridge = create_bridge()
            assert isinstance(bridge, ClaudeBridge)

    def test_gemini(self):
        from unittest.mock import patch
        with patch("web.backend.agent_bridge.cfg") as mock_cfg:
            mock_cfg.agent_provider = "gemini"
            bridge = create_bridge()
            assert isinstance(bridge, GeminiBridge)

    def test_codex(self):
        from unittest.mock import patch
        with patch("web.backend.agent_bridge.cfg") as mock_cfg:
            mock_cfg.agent_provider = "codex"
            bridge = create_bridge()
            assert isinstance(bridge, CodexBridge)

    def test_unknown_provider_raises(self):
        from unittest.mock import patch
        with patch("web.backend.agent_bridge.cfg") as mock_cfg:
            mock_cfg.agent_provider = "unknown"
            with pytest.raises(ValueError, match="Unknown agent provider"):
                create_bridge()
