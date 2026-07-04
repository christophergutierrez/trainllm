#!/usr/bin/env python3
"""
Sandboxed execution for HumanEval-style Python code.

HumanEval format: run (prompt + completion + test_code) as a single script
and check that all assertions pass within a timeout.

Usage:
    from code_verify import verify_humaneval
    result = verify_humaneval(prompt, completion, test_code)
    print(result.passed, result.stderr)
"""

from __future__ import annotations

import resource
import subprocess
import sys
import re
from dataclasses import dataclass

_MEMORY_LIMIT = 4 * 1024 * 1024 * 1024   # 4 GB — numpy/OpenBLAS needs headroom
_CPU_LIMIT    = 10                   # CPU seconds
DEFAULT_TIMEOUT = 10.0               # wall-clock seconds


@dataclass
class VerifyResult:
    passed: bool
    stderr: str = ""
    timed_out: bool = False


def clean_completion(text: str) -> str:
    """Normalize model output before executing it as Python."""
    import re

    stripped = text.strip()
    fenced = re.match(r"^```(?:python|py)?\n(.*?)(?:\n?```.*)?$", stripped, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return re.split(r"\n```", text, maxsplit=1)[0].strip("\n")


def _set_limits() -> None:
    resource.setrlimit(resource.RLIMIT_AS,  (_MEMORY_LIMIT, _MEMORY_LIMIT))
    resource.setrlimit(resource.RLIMIT_CPU, (_CPU_LIMIT,    _CPU_LIMIT))


def _script_under_test(prompt: str, completion: str, test_code: str) -> str:
    completion = clean_completion(completion)
    if prompt.endswith("\n"):
        solution = prompt + completion
    else:
        solution = prompt + "\n" + completion
    return solution + "\n\n" + test_code


def verify_humaneval(
    prompt: str,
    completion: str,
    test_code: str,
    timeout: float = DEFAULT_TIMEOUT,
    entry_point: str = "",
) -> VerifyResult:
    """Run prompt + completion + test harness in a sandbox, return pass/fail.

    HumanEval test_code typically looks like:
        def check(candidate):
            assert candidate(1) == 2
        check(<entry_point>)
    """
    # HumanEval+ defines check() but doesn't call it; append the call if missing
    tail = test_code.rstrip()
    has_check_call = re.search(r"(?m)^\s*check\s*\(", tail) is not None
    if entry_point and not has_check_call:
        tail = tail + f"\ncheck({entry_point})"
    script = _script_under_test(prompt, completion, tail)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout,
            preexec_fn=_set_limits,
        )
        if proc.returncode == 0:
            return VerifyResult(passed=True)
        return VerifyResult(passed=False, stderr=proc.stderr[:500])
    except subprocess.TimeoutExpired:
        return VerifyResult(passed=False, timed_out=True, stderr="timeout")
    except Exception as e:
        return VerifyResult(passed=False, stderr=str(e)[:200])
