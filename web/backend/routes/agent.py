"""REST endpoints for agent interaction — file-based relay to Claude Code session."""

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from ..ws import manager, Channel
from ..agent_bridge import is_agent_available, write_to_inbox, _message_log

router = APIRouter()


class AgentCommand(BaseModel):
    content: str
    context: Optional[dict] = None


@router.post("/send")
async def send_command(cmd: AgentCommand):
    """Write a message to the agent inbox. The Claude Code session polls this."""
    if not is_agent_available():
        return {"status": "disabled", "error": "No agent session active"}

    msg_id = await write_to_inbox(cmd.content, cmd.context)
    await manager.broadcast(Channel.AGENT, {
        "type": "user_command",
        "content": cmd.content,
        "id": msg_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    return {"status": "sent", "id": msg_id}


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
    """Check if the agent is available (presence file fresh) and WS client count."""
    return {
        "available": is_agent_available(),
        "ws_clients": manager.client_count(Channel.AGENT),
        "message_count": len(_message_log),
    }
