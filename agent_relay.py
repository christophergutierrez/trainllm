#!/usr/bin/env python3
"""Agent relay — keeps presence alive and polls inbox for messages.

This script is meant to run in the background during a Claude Code session.
It refreshes the presence file every 60s and writes new inbox messages
to stdout (one JSON per line) so the calling process can read them.

Usage:
    python agent_relay.py              # foreground, prints inbox messages to stdout
    python agent_relay.py --presence   # only refresh presence (no inbox polling)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

PRESENCE_FILE = Path(os.environ.get("TRAINLLM_AGENT_PRESENCE", "/tmp/trainllm_agent_presence.json"))
INBOX_FILE = Path(os.environ.get("TRAINLLM_AGENT_INBOX", "/tmp/trainllm_agent_inbox.jsonl"))
OUTBOX_FILE = Path(os.environ.get("TRAINLLM_AGENT_OUTBOX", "/tmp/trainllm_agent_outbox.jsonl"))

REFRESH_INTERVAL = 60


def refresh_presence():
    """Write/update the presence file with current PID and timestamp."""
    data = {"pid": os.getpid(), "timestamp": time.time(), "session": "claude-code"}
    PRESENCE_FILE.write_text(json.dumps(data))


def write_response(msg_id: str, content: str):
    """Write a response to the outbox file."""
    from datetime import datetime, timezone
    entry = {
        "id": msg_id,
        "content": content,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(OUTBOX_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def poll_inbox():
    """Yield new inbox messages as they arrive."""
    pos = 0
    if INBOX_FILE.exists():
        pos = INBOX_FILE.stat().st_size

    last_refresh = 0
    while True:
        now = time.time()
        if now - last_refresh >= REFRESH_INTERVAL:
            refresh_presence()
            last_refresh = now

        if not INBOX_FILE.exists():
            pos = 0
            time.sleep(1)
            continue

        size = INBOX_FILE.stat().st_size
        if size < pos:
            pos = 0
        if size == pos:
            time.sleep(1)
            continue

        with open(INBOX_FILE) as f:
            f.seek(pos)
            for line in f:
                line = line.strip()
                if line:
                    try:
                        msg = json.loads(line)
                        yield msg
                    except json.JSONDecodeError:
                        pass
            pos = f.tell()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--presence", action="store_true", help="Only refresh presence, don't poll inbox")
    args = parser.parse_args()

    refresh_presence()
    print(f"Presence written: {PRESENCE_FILE}", file=sys.stderr)

    if args.presence:
        try:
            while True:
                time.sleep(REFRESH_INTERVAL)
                refresh_presence()
        except KeyboardInterrupt:
            pass
        return

    print(f"Polling inbox: {INBOX_FILE}", file=sys.stderr)
    print(f"Outbox: {OUTBOX_FILE}", file=sys.stderr)

    try:
        for msg in poll_inbox():
            print(json.dumps(msg), flush=True)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
