"""REST endpoints for agent interaction — allows external agents to send/receive via HTTP."""

from collections import deque
from datetime import datetime, timezone

from fastapi import APIRouter
from pydantic import BaseModel

from ..ws import manager, Channel

router = APIRouter()

_message_log: deque = deque(maxlen=100)


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

    # Trigger the bridge response
    from ..agent_bridge import create_bridge
    bridge = create_bridge()
    if not bridge.is_alive:
        await bridge.spawn()
    await bridge.send(cmd.content)

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
    }
