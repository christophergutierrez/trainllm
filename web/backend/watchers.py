"""File watchers that push pipeline events to WebSocket clients."""

import asyncio
import json
from pathlib import Path

from .config import cfg
from .ws import manager, Channel


async def watch_events_file():
    """Tail the training events JSONL and broadcast new lines."""
    path = cfg.events_file
    last_pos = 0
    if path.exists():
        last_pos = path.stat().st_size

    while True:
        await asyncio.sleep(1)
        if not path.exists():
            continue
        size = path.stat().st_size
        if size <= last_pos:
            if size < last_pos:
                last_pos = 0
            continue
        with open(path) as f:
            f.seek(last_pos)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    await manager.broadcast(Channel.TRAINING, event)
                except json.JSONDecodeError:
                    pass
            last_pos = f.tell()


async def watch_recommendations():
    """Watch recommendations.yaml and broadcast changes to agent channel."""
    path = cfg.recommendations_file
    last_mtime = 0.0

    while True:
        await asyncio.sleep(2)
        if not path.exists():
            continue
        mtime = path.stat().st_mtime
        if mtime <= last_mtime:
            continue
        last_mtime = mtime
        try:
            import yaml
            recs = yaml.safe_load(path.read_text()) or []
            await manager.broadcast(Channel.AGENT, {
                "type": "recommendations",
                "data": recs,
            })
        except Exception:
            pass


def start_watchers(loop: asyncio.AbstractEventLoop):
    """Start all file watcher tasks."""
    loop.create_task(watch_events_file())
    loop.create_task(watch_recommendations())
