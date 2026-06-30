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


def _lean_cmd(path: Path, project_dir: Path | None) -> list[str]:
    """
    Return the command list to compile a .lean file.

    With project_dir set, runs `lake env lean <file>` from the project root so
    the SLT and Mathlib oleans are on LEAN_PATH.  Without it, falls back to a
    bare `lean <file>` call (only stdlib available).
    """
    lean = _find_lean()
    if lean is None:
        return []
    if project_dir is not None and (project_dir / "lakefile.lean").exists():
        return ["lake", "env", "lean", str(path)]
    return [lean, str(path)]


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


_import_header_cache: dict[str, str] = {}


def _build_import_header(project_dir: Path) -> str:
    """
    Build an import + open header for all modules in the project's main library,
    plus Mathlib.  Also opens all SLT namespaces so unqualified identifiers from
    the original source files resolve in standalone example blocks.
    Results are cached by project path.
    """
    key = str(project_dir)
    if key in _import_header_cache:
        return _import_header_cache[key]

    # Top-level Mathlib namespaces that the SLT source files open.
    # Kept as a hardcoded allowlist — auto-discovery of nested SLT namespaces
    # (e.g. LeastSquares.EmpiricalSpace) causes "unknown namespace" errors.
    SAFE_OPENS = [
        "MeasureTheory", "ProbabilityTheory", "Real", "Set", "Filter",
        "Function", "Finset", "Metric", "Classical", "Nat", "Complex",
        "TopologicalSpace", "BoundedContinuousFunction", "Convolution",
        "EReal", "RealInnerProductSpace",
    ]
    SAFE_SCOPED = ["ENNReal", "NNReal", "BigOperators", "Topology", "Pointwise"]

    import_lines = ["import Mathlib"]
    slt_namespaces: list[str] = []

    # Find the library root: first capitalised subdirectory (SLT/ for this project).
    for sub in sorted(project_dir.iterdir()):
        if sub.is_dir() and not sub.name.startswith(".") and sub.name[0].isupper():
            for f in sorted(sub.rglob("*.lean")):
                module = ".".join(f.relative_to(project_dir).parts).removesuffix(".lean")
                import_lines.append(f"import {module}")
                # Collect top-level namespace names (depth 0 only).
                try:
                    depth = 0
                    for ln in f.read_text(errors="replace").splitlines():
                        s = ln.strip()
                        if s.startswith("namespace ") and not s.startswith("namespace where"):
                            if depth == 0:
                                ns = s.split()[1]
                                if ns not in slt_namespaces:
                                    slt_namespaces.append(ns)
                            depth += 1
                        elif s.startswith("end ") and depth > 0:
                            depth -= 1
                except Exception:
                    pass
            break

    all_opens = SAFE_OPENS + [ns for ns in slt_namespaces if ns not in SAFE_OPENS]
    lines = import_lines + [
        "",
        f"open {' '.join(all_opens)}",
        f"open scoped {' '.join(SAFE_SCOPED)}",
        "",
    ]

    header = "\n".join(lines) + "\n"
    _import_header_cache[key] = header
    return header


def _make_lean_snippet(state_before: str, tactic: str,
                       header: str = "") -> str | None:
    """
    Build a minimal standalone Lean 4 `example` block from a goal state + tactic.

    header: optional import lines prepended before the example (for project-aware runs).
    Returns None if the state cannot be reconstructed as a standalone theorem.
    """
    hyps, goal = _parse_state(state_before)
    if goal is None:
        return None
    hyp_str = " ".join(f"({h})" for h in hyps)
    return f"{header}example {hyp_str}: {goal} := by\n  {tactic}\n"


def verify_tactic(
    state_before: str,
    tactic: str,
    timeout: int = 60,
    project_dir: Path | None = None,
) -> VerifyResult:
    """
    Safety-check then Lean-verify a tactic against a goal state.

    If the state cannot be parsed into a standalone snippet, or if lean is not
    installed, lean_ok is set to None and passed reflects safety only.

    project_dir: path to a Lake project root (e.g. lean-stat-learning-theory).
    When set, compilation runs via `lake env lean` so SLT/Mathlib oleans are
    available.  Without it, only Lean stdlib is accessible.
    """
    if not safety_check(tactic):
        return VerifyResult(passed=False, safety_ok=False, lean_ok=None,
                            stdout="", stderr="forbidden token in tactic", elapsed=0.0)

    header = _build_import_header(project_dir) if project_dir else ""
    snippet = _make_lean_snippet(state_before, tactic, header=header)
    if snippet is None:
        return VerifyResult(passed=False, safety_ok=True, lean_ok=None,
                            stdout="",
                            stderr="state cannot be reconstructed as standalone example",
                            elapsed=0.0)

    if _find_lean() is None:
        return VerifyResult(passed=False, safety_ok=True, lean_ok=None,
                            stdout="", stderr="lean executable not found in PATH",
                            elapsed=0.0)

    with tempfile.NamedTemporaryFile(suffix=".lean", mode="w",
                                     delete=False) as f:
        f.write(snippet)
        tmp = Path(f.name)
    try:
        return verify_file(tmp, timeout=timeout, project_dir=project_dir)
    finally:
        tmp.unlink(missing_ok=True)


def verify_file(path: Path, timeout: int = 60,
                project_dir: Path | None = None) -> VerifyResult:
    """
    Run the Lean compiler on an existing .lean file and return the result.

    project_dir: when set, runs `lake env lean <file>` from that directory so
    project-local and Mathlib identifiers resolve.
    """
    cmd = _lean_cmd(path, project_dir)
    if not cmd:
        return VerifyResult(passed=False, safety_ok=True, lean_ok=None,
                            stdout="", stderr="lean executable not found in PATH",
                            elapsed=0.0)
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(project_dir) if project_dir else None,
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
                            stdout="", stderr=f"lean not found: {cmd[0]}",
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
    p.add_argument("--lean-project", metavar="DIR", type=Path, default=None,
                   help="Lake project root — enables `lake env lean` so SLT/Mathlib "
                        "identifiers resolve (e.g. ~/git_home/lean-stat-learning-theory)")
    args = p.parse_args()

    if args.check is not None:
        ok = safety_check(args.check)
        print("SAFE" if ok else "FORBIDDEN")
        sys.exit(0 if ok else 1)

    if args.file is not None:
        result = verify_file(args.file, timeout=args.timeout,
                             project_dir=args.lean_project)
    else:
        if not args.state:
            p.error("--state is required when using --tactic")
        result = verify_tactic(args.state, args.tactic, timeout=args.timeout,
                               project_dir=args.lean_project)

    import json
    print(json.dumps(result.to_dict(), indent=2))
    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
