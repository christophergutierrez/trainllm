"""WebSocket manager for real-time training events and agent communication."""

import asyncio
import json
from enum import Enum
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()


class Channel(str, Enum):
    TRAINING = "training"
    AGENT = "agent"


class ConnectionManager:
    def __init__(self):
        self._connections: dict[Channel, list[WebSocket]] = {
            Channel.TRAINING: [],
            Channel.AGENT: [],
        }
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket, channel: Channel):
        await ws.accept()
        async with self._lock:
            self._connections[channel].append(ws)

    async def disconnect(self, ws: WebSocket, channel: Channel):
        async with self._lock:
            if ws in self._connections[channel]:
                self._connections[channel].remove(ws)

    async def broadcast(self, channel: Channel, data: dict[str, Any]):
        message = json.dumps(data)
        async with self._lock:
            stale = []
            for ws in self._connections[channel]:
                try:
                    await ws.send_text(message)
                except Exception:
                    stale.append(ws)
            for ws in stale:
                self._connections[channel].remove(ws)

    def client_count(self, channel: Channel) -> int:
        return len(self._connections[channel])


manager = ConnectionManager()


@router.websocket("/ws/training")
async def ws_training(ws: WebSocket):
    await manager.connect(ws, Channel.TRAINING)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await manager.disconnect(ws, Channel.TRAINING)


@router.websocket("/ws/agent")
async def ws_agent(ws: WebSocket):
    await manager.connect(ws, Channel.AGENT)
    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            await manager.broadcast(Channel.AGENT, {
                "type": "user_command",
                "content": msg.get("content", ""),
            })
    except WebSocketDisconnect:
        pass
    finally:
        await manager.disconnect(ws, Channel.AGENT)
