#!/usr/bin/env python3
"""
Lean 4 tactic verification harness.

Two levels of checking:
  1. Safety check (pure Python) — rejects sorry/admit regardless of Lean availability.
  2. Lean compilation — inserts the tactic into a temporary file and invokes lean.

Usage:
  python lean_verify.py --tactic "simp" --state "n : Nat\\n⊢ n + 0 = n"
  python lean_verify.py --check "simp"          # safety-check only, no Lean needed
  python lean_verify.py --file path/to/file.lean  # verify a pre-written .lean file
"""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

FORBIDDEN = re.compile(r"\b(sorry|admit)\b|#\s*check\b")


@dataclass
class VerifyResult:
    passed: bool
    safety_ok: bool
    lean_ok: bool | None   # None = lean not run (not found or safety failed)
    stdout: str
    stderr: str
    elapsed: float

    def to_dict(self) -> dict:
        return asdict(self)


def safety_check(tactic: str) -> bool:
    """Return True if the tactic contains no forbidden tokens."""
    return not FORBIDDEN.search(tactic)


def _find_lean() -> str | None:
    return shutil.which("lean") or shutil.which("lake")


def _parse_state(state_before: str) -> tuple[list[str], str] | tuple[None, None]:
    """
    Parse a Lean 4 tactic state into (hypotheses, goal).

    State format:
        h1 : Type1
        h2 : Type2
        ⊢ goal_expr

    Returns (hyps, goal) or (None, None) if the goal line is missing.
    Only parses states simple enough to reconstruct as a standalone example.
    Rejects states with universe metavariables (?u), expression holes (?m),
    or multiline goals.
    """
    hyps: list[str] = []
    goal: str | None = None
    for raw in state_before.strip().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("⊢"):
            goal = line[1:].strip()
        elif re.match(r"^\w[\w✝ ]*:", line):
            hyps.append(line)
    if goal is None:
        return None, None
    # Reject states that reference metavariables or instance dummies we can't reconstruct.
    if re.search(r"\?[mu]\w*|inst✝", goal + " ".join(hyps)):
        return None, None
    return hyps, goal


def _make_lean_snippet(state_before: str, tactic: str) -> str | None:
    """
    Build a minimal standalone Lean 4 `example` block from a goal state + tactic.

    Returns None if the state cannot be reconstructed as a standalone theorem.
    """
    hyps, goal = _parse_state(state_before)
    if goal is None:
        return None
    hyp_str = " ".join(f"({h})" for h in hyps)
    return f"example {hyp_str}: {goal} := by\n  {tactic}\n"


def verify_tactic(
    state_before: str,
    tactic: str,
    timeout: int = 60,
) -> VerifyResult:
    """
    Safety-check then Lean-verify a tactic against a goal state.

    If the state cannot be parsed into a standalone snippet, or if lean is not
    installed, lean_ok is set to None and passed reflects safety only.
    """
    if not safety_check(tactic):
        return VerifyResult(passed=False, safety_ok=False, lean_ok=None,
                            stdout="", stderr="forbidden token in tactic", elapsed=0.0)

    snippet = _make_lean_snippet(state_before, tactic)
    if snippet is None:
        return VerifyResult(passed=False, safety_ok=True, lean_ok=None,
                            stdout="",
                            stderr="state cannot be reconstructed as standalone example",
                            elapsed=0.0)

    lean = _find_lean()
    if lean is None:
        return VerifyResult(passed=False, safety_ok=True, lean_ok=None,
                            stdout="", stderr="lean executable not found in PATH",
                            elapsed=0.0)

    with tempfile.NamedTemporaryFile(suffix=".lean", mode="w",
                                     delete=False) as f:
        f.write(snippet)
        tmp = Path(f.name)
    try:
        return verify_file(tmp, timeout=timeout)
    finally:
        tmp.unlink(missing_ok=True)


def verify_file(path: Path, timeout: int = 60) -> VerifyResult:
    """Run the Lean compiler on an existing .lean file and return the result."""
    lean = _find_lean()
    if lean is None:
        return VerifyResult(passed=False, safety_ok=True, lean_ok=None,
                            stdout="", stderr="lean executable not found in PATH",
                            elapsed=0.0)
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            [lean, str(path)],
            capture_output=True, text=True, timeout=timeout,
        )
        elapsed = time.monotonic() - t0
        ok = proc.returncode == 0
        return VerifyResult(passed=ok, safety_ok=True, lean_ok=ok,
                            stdout=proc.stdout, stderr=proc.stderr, elapsed=elapsed)
    except subprocess.TimeoutExpired:
        return VerifyResult(passed=False, safety_ok=True, lean_ok=False,
                            stdout="", stderr="lean timed out",
                            elapsed=float(timeout))
    except FileNotFoundError:
        return VerifyResult(passed=False, safety_ok=True, lean_ok=None,
                            stdout="", stderr=f"lean not found: {lean}",
                            elapsed=0.0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--tactic", help="Tactic string to verify against --state")
    g.add_argument("--check", metavar="TACTIC",
                   help="Safety-check only (no Lean compilation)")
    g.add_argument("--file", metavar="PATH", type=Path,
                   help="Verify a pre-written .lean file")
    p.add_argument("--state", help="Lean 4 tactic state (required with --tactic)")
    p.add_argument("--timeout", type=int, default=60)
    args = p.parse_args()

    if args.check is not None:
        ok = safety_check(args.check)
        print("SAFE" if ok else "FORBIDDEN")
        sys.exit(0 if ok else 1)

    if args.file is not None:
        result = verify_file(args.file, timeout=args.timeout)
    else:
        if not args.state:
            p.error("--state is required when using --tactic")
        result = verify_tactic(args.state, args.tactic, timeout=args.timeout)

    import json
    print(json.dumps(result.to_dict(), indent=2))
    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
