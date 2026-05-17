"""REST endpoints for agent interaction — allows external agents to send/receive via HTTP."""

import asyncio
from datetime import datetime, timezone

from fastapi import APIRouter
from pydantic import BaseModel

from ..ws import manager, Channel
from ..agent_bridge import get_bridge, _message_log

router = APIRouter()


class AgentCommand(BaseModel):
    content: str


@router.post("/send")
async def send_command(cmd: AgentCommand):
    """Send a command to the agent (same as typing in the UI and hitting submit)."""
    msg = {
        "type": "user_command",
        "content": cmd.content,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _message_log.append(msg)
    await manager.broadcast(Channel.AGENT, msg)

    bridge = get_bridge()
    if not bridge.is_alive:
        await bridge.spawn()

    # Fire and forget — response streams back via WebSocket
    asyncio.create_task(bridge.send(cmd.content))

    return {"status": "sent", "content": cmd.content}


@router.get("/messages")
async def get_messages(since: int = 0, limit: int = 50):
    """Get recent agent messages. Use `since` to get only new messages (index-based)."""
    messages = list(_message_log)
    if since > 0:
        messages = messages[since:]
    return {
        "messages": messages[:limit],
        "total": len(_message_log),
        "next_since": len(_message_log),
    }


@router.get("/status")
async def agent_status():
    """Check if the agent bridge is alive and how many WS clients are connected."""
    return {
        "ws_clients": manager.client_count(Channel.AGENT),
        "message_count": len(_message_log),
        "bridge_alive": get_bridge().is_alive,
    }
