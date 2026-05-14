#!/usr/bin/env python3
"""Find and kill all vLLM processes and their children (EngineCore, NCCL, resource_tracker).

Usage:
    python kill_vllm.py          # dry-run: show what would be killed
    python kill_vllm.py --kill   # actually kill everything
    python kill_vllm.py --kill --verify  # kill, wait, confirm cleanup

vLLM spawns child processes that survive parent SIGTERM/SIGKILL:
  - vllm.v1.engine.core (EngineCore worker)
  - nccl_heartbeat_monitor
  - multiprocessing.resource_tracker
These hold GPU memory and cause OOM if not cleaned up before training.
"""

import argparse
import os
import signal
import subprocess
import sys
import time

VLLM_PATTERNS = [
    "vllm serve",
    "vllm.entrypoints",
    "vllm.v1.engine.core",
    "nccl_heartbeat_monitor",
    "multiprocessing.resource_tracker",
]


def find_vllm_procs() -> list[dict]:
    """Return list of {pid, ppid, cmd} for all vLLM-related processes."""
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid,ppid,args", "--no-headers"],
            capture_output=True, text=True,
        )
    except FileNotFoundError:
        return []

    procs = []
    for line in result.stdout.strip().splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid, ppid, cmd = parts[0], parts[1], parts[2]
        if any(pat in cmd for pat in VLLM_PATTERNS):
            if "kill_vllm.py" in cmd or "grep" in cmd:
                continue
            procs.append({"pid": int(pid), "ppid": int(ppid), "cmd": cmd[:120]})
    return procs


def kill_process_tree(pid: int, sig: int = signal.SIGTERM) -> list[int]:
    """Kill a process and all its descendants via process group."""
    killed = []
    try:
        pgid = os.getpgid(pid)
        os.killpg(pgid, sig)
        killed.append(pid)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, sig)
            killed.append(pid)
        except (ProcessLookupError, PermissionError):
            pass
    return killed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kill", action="store_true", help="Actually kill processes (default is dry-run)")
    parser.add_argument("--force", action="store_true", help="SIGKILL after SIGTERM fails (leaks GPU memory on GB10)")
    parser.add_argument("--verify", action="store_true", help="After kill, wait and verify cleanup")
    args = parser.parse_args()

    procs = find_vllm_procs()
    if not procs:
        print("No vLLM processes found. System is clean.")
        return 0

    print(f"Found {len(procs)} vLLM-related process(es):")
    for p in procs:
        print(f"  PID {p['pid']:>7}  PPID {p['ppid']:>7}  {p['cmd']}")

    if not args.kill:
        print("\nDry run — pass --kill to terminate these processes.")
        return 1

    seen_pgids = set()
    killed_pids = set()
    for p in procs:
        try:
            pgid = os.getpgid(p["pid"])
        except (ProcessLookupError, PermissionError):
            continue
        if pgid not in seen_pgids:
            seen_pgids.add(pgid)
            try:
                os.killpg(pgid, signal.SIGTERM)
                print(f"  SIGTERM → process group {pgid}")
            except (ProcessLookupError, PermissionError):
                pass
        killed_pids.add(p["pid"])

    # GB10 unified memory: SIGKILL leaks CUDA driver memory permanently.
    # Give vLLM 30s to clean up CUDA contexts before escalating.
    print("  Waiting up to 30s for clean CUDA teardown...")
    for i in range(30):
        time.sleep(1)
        remaining = find_vllm_procs()
        if not remaining:
            print(f"  All processes exited cleanly after {i+1}s.")
            break
    else:
        remaining = find_vllm_procs()

    if remaining:
        print(f"\n  WARNING: {len(remaining)} process(es) survived 30s SIGTERM.")
        print("  SIGKILL will leak GPU memory on GB10 unified memory.")
        print("  You may need to reload NVIDIA modules: sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia && sudo modprobe nvidia nvidia_uvm nvidia_drm nvidia_modeset")
        if not args.force:
            print("  Pass --force to SIGKILL anyway.")
            return 2
        print("  --force specified, sending SIGKILL:")
        seen_pgids.clear()
        for p in remaining:
            try:
                pgid = os.getpgid(p["pid"])
            except (ProcessLookupError, PermissionError):
                try:
                    os.kill(p["pid"], signal.SIGKILL)
                    print(f"  SIGKILL → PID {p['pid']}")
                except (ProcessLookupError, PermissionError):
                    pass
                continue
            if pgid not in seen_pgids:
                seen_pgids.add(pgid)
                try:
                    os.killpg(pgid, signal.SIGKILL)
                    print(f"  SIGKILL → process group {pgid}")
                except (ProcessLookupError, PermissionError):
                    pass
        time.sleep(2)

    if args.verify:
        final = find_vllm_procs()
        if final:
            print(f"\nWARNING: {len(final)} process(es) still alive after SIGKILL:")
            for p in final:
                print(f"  PID {p['pid']:>7}  {p['cmd']}")
            return 2
        print("\nVerified: all vLLM processes cleaned up.")

    remaining = find_vllm_procs()
    if not remaining:
        print(f"\nKilled {len(killed_pids)} process(es). System is clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
