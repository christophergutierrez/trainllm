"""Abstract agent bridge and concrete implementations.

The bridge manages a long-running AI agent process that observes pipeline state
and accepts user commands. The frontend is completely agent-agnostic — it sends
and receives text via WebSocket, and the bridge handles provider-specific translation.
"""

import asyncio
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime, timezone
from typing import AsyncIterator

from .config import cfg
from .ws import manager, Channel

# Shared message log — used by both the bridge and the REST routes
_message_log: deque = deque(maxlen=100)


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
    """Anthropic Messages API bridge with streaming."""

    _SYSTEM = (
        "You are an AI assistant embedded in the trainLLM dashboard — a tool for "
        "fine-tuning language models on structured API call generation tasks. "
        "You help users interpret training runs, debug pipeline issues, understand "
        "evaluation metrics, and plan training cycles.\n\n"
        "Evaluation bands: GOLD > SILVER > BRONZE > FAIL. "
        "composite_score combines structural correctness and semantic accuracy. "
        "structural_score measures JSON format validity and parameter completeness.\n\n"
        "Be concise and technical. The user is an ML engineer."
    )

    def __init__(self):
        self._client = None
        self._alive = False
        self._history: list[dict] = []

    async def spawn(self) -> None:
        import anthropic
        self._client = anthropic.AsyncAnthropic()
        self._alive = True
        await manager.broadcast(Channel.AGENT, {
            "type": "agent_status",
            "status": "connected",
            "provider": "claude",
        })

    async def send(self, message: str) -> None:
        if not self._client:
            await self.spawn()

        self._history.append({"role": "user", "content": message})

        full_response = ""
        try:
            async with self._client.messages.stream(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                system=self._SYSTEM,
                messages=self._history,
            ) as stream:
                async for chunk in stream.text_stream:
                    full_response += chunk
                    await manager.broadcast(Channel.AGENT, {
                        "type": "agent_chunk",
                        "content": chunk,
                    })
        except Exception as exc:
            self._history.pop()
            error = f"[Claude error: {exc}]"
            entry = {
                "type": "agent_response",
                "content": error,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            _message_log.append(entry)
            await manager.broadcast(Channel.AGENT, entry)
            return

        self._history.append({"role": "assistant", "content": full_response})
        entry = {
            "type": "agent_response",
            "content": full_response,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        _message_log.append(entry)
        await manager.broadcast(Channel.AGENT, {
            "type": "agent_response_end",
            "content": full_response,
            "timestamp": entry["timestamp"],
        })

    async def receive(self) -> AsyncIterator[str]:
        while self._alive:
            await asyncio.sleep(1)
            yield ""

    async def shutdown(self) -> None:
        self._alive = False
        if self._client:
            await self._client.close()
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

_bridge: AgentBridge | None = None


def create_bridge() -> AgentBridge:
    """Create a new agent bridge for the configured provider."""
    provider = cfg.agent_provider
    bridge_cls = _BRIDGES.get(provider)
    if not bridge_cls:
        raise ValueError(f"Unknown agent provider: {provider}. Options: {list(_BRIDGES.keys())}")
    return bridge_cls()


def get_bridge() -> AgentBridge:
    """Return the singleton agent bridge, creating it if needed."""
    global _bridge
    if _bridge is None:
        _bridge = create_bridge()
    return _bridge
