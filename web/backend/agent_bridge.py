"""File-based agent relay — writes to inbox, polls outbox, checks presence."""

import asyncio
import json
import time
import uuid
from collections import deque
from datetime import datetime, timezone

from .config import cfg
from .ws import manager, Channel

_message_log: deque = deque(maxlen=100)
_outbox_task: asyncio.Task | None = None


def is_agent_available() -> bool:
    """Check if a Claude Code session is active (presence file fresh)."""
    p = cfg.agent_presence
    if not p.exists():
        return False
    try:
        data = json.loads(p.read_text())
        ts = data.get("timestamp", 0)
        return (time.time() - ts) < cfg.agent_presence_ttl
    except (json.JSONDecodeError, OSError):
        return False


async def write_to_inbox(message: str, context: dict | None = None) -> str:
    """Append a user message to the inbox file. Returns the message ID."""
    msg_id = str(uuid.uuid4())[:8]
    entry = {
        "id": msg_id,
        "message": message,
        "context": context or {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(cfg.agent_inbox, "a") as f:
        f.write(json.dumps(entry) + "\n")

    _message_log.append({"type": "user_command", "content": message, "id": msg_id})
    return msg_id


async def poll_outbox():
    """Background task: tail the outbox file and broadcast responses."""
    path = cfg.agent_outbox
    pos = 0
    if path.exists():
        pos = path.stat().st_size

    while True:
        await asyncio.sleep(1)
        if not path.exists():
            pos = 0
            continue
        size = path.stat().st_size
        if size < pos:
            pos = 0
        if size == pos:
            continue
        try:
            with open(path) as f:
                f.seek(pos)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                        body = msg.get("content") or msg.get("response") or ""
                        entry = {
                            "type": "agent_response",
                            "content": body,
                            "id": msg.get("id", ""),
                            "timestamp": msg.get("timestamp", ""),
                        }
                        _message_log.append(entry)
                        await manager.broadcast(Channel.AGENT, entry)
                    except json.JSONDecodeError:
                        pass
                pos = f.tell()
        except OSError:
            pass


def start_outbox_poller():
    """Start the outbox polling background task."""
    global _outbox_task
    if _outbox_task is None or _outbox_task.done():
        _outbox_task = asyncio.create_task(poll_outbox())
    return _outbox_task
