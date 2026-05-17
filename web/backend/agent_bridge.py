"""Abstract agent bridge and concrete implementations.

The bridge manages a long-running AI agent process that observes pipeline state
and accepts user commands. The frontend is completely agent-agnostic — it sends
and receives text via WebSocket, and the bridge handles provider-specific translation.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import AsyncIterator

from .config import cfg
from .ws import manager, Channel


class AgentBridge(ABC):
    """Abstract interface for AI agent communication."""

    @abstractmethod
    async def spawn(self) -> None:
        """Start the agent process."""

    @abstractmethod
    async def send(self, message: str) -> None:
        """Send a user message/command to the agent."""

    @abstractmethod
    async def receive(self) -> AsyncIterator[str]:
        """Yield agent responses as they arrive."""

    @abstractmethod
    async def shutdown(self) -> None:
        """Gracefully stop the agent."""

    @property
    @abstractmethod
    def is_alive(self) -> bool:
        """Whether the agent process is running."""


class ClaudeBridge(AgentBridge):
    """Claude Agent SDK bridge."""

    def __init__(self):
        self._process = None
        self._alive = False

    async def spawn(self) -> None:
        self._alive = True
        await manager.broadcast(Channel.AGENT, {
            "type": "agent_status",
            "status": "connected",
            "provider": "claude",
        })

    async def send(self, message: str) -> None:
        await manager.broadcast(Channel.AGENT, {
            "type": "agent_thinking",
            "content": f"Processing: {message}",
        })

    async def receive(self) -> AsyncIterator[str]:
        while self._alive:
            await asyncio.sleep(1)
            yield ""

    async def shutdown(self) -> None:
        self._alive = False
        await manager.broadcast(Channel.AGENT, {
            "type": "agent_status",
            "status": "disconnected",
        })

    @property
    def is_alive(self) -> bool:
        return self._alive


class GeminiBridge(AgentBridge):
    """Gemini CLI / google-genai bridge (placeholder)."""

    def __init__(self):
        self._alive = False

    async def spawn(self) -> None:
        self._alive = True
        await manager.broadcast(Channel.AGENT, {
            "type": "agent_status",
            "status": "connected",
            "provider": "gemini",
        })

    async def send(self, message: str) -> None:
        await manager.broadcast(Channel.AGENT, {
            "type": "agent_response",
            "content": "[Gemini bridge not yet implemented]",
        })

    async def receive(self) -> AsyncIterator[str]:
        while self._alive:
            await asyncio.sleep(1)
            yield ""

    async def shutdown(self) -> None:
        self._alive = False

    @property
    def is_alive(self) -> bool:
        return self._alive


class CodexBridge(AgentBridge):
    """OpenAI Codex CLI bridge (placeholder)."""

    def __init__(self):
        self._alive = False

    async def spawn(self) -> None:
        self._alive = True
        await manager.broadcast(Channel.AGENT, {
            "type": "agent_status",
            "status": "connected",
            "provider": "codex",
        })

    async def send(self, message: str) -> None:
        await manager.broadcast(Channel.AGENT, {
            "type": "agent_response",
            "content": "[Codex bridge not yet implemented]",
        })

    async def receive(self) -> AsyncIterator[str]:
        while self._alive:
            await asyncio.sleep(1)
            yield ""

    async def shutdown(self) -> None:
        self._alive = False

    @property
    def is_alive(self) -> bool:
        return self._alive


_BRIDGES = {
    "claude": ClaudeBridge,
    "gemini": GeminiBridge,
    "codex": CodexBridge,
}


def create_bridge() -> AgentBridge:
    """Create the configured agent bridge."""
    provider = cfg.agent_provider
    bridge_cls = _BRIDGES.get(provider)
    if not bridge_cls:
        raise ValueError(f"Unknown agent provider: {provider}. Options: {list(_BRIDGES.keys())}")
    return bridge_cls()
