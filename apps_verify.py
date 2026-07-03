#!/usr/bin/env python3
"""
Safety scan and sandboxed execution for APPS-generated Python code.

Two stages:
  1. safety_scan(code)     — regex scan for dangerous patterns; fast, no execution
  2. execute_solution(...) — run in a child process with timeout + resource limits

Supports both APPS test formats:
  stdio   {"inputs": ["5\\n"], "outputs": ["25\\n"]}
  fn_call {"fn_name": "twoSum", "inputs": [[2,7], 9], "outputs": [[0,1]]}

Usage (as a library):
    from apps_verify import safety_scan, execute_solution
    scan = safety_scan(code)
    if scan.safe:
        result = execute_solution(code, problem["input_output"])
"""

from __future__ import annotations

import json
import re
import resource
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field

# ── Safety scan ──────────────────────────────────────────────────────────────

# (pattern, reason) — matched against the raw source text
_BLOCKED: list[tuple[str, str]] = [
    (r"\bsubprocess\b",                  "subprocess"),
    (r"\bos\.system\s*\(",               "os.system"),
    (r"\bos\.popen\s*\(",                "os.popen"),
    (r"\bos\.(exec|spawn)\w*\s*\(",      "os.exec/spawn"),
    (r"import\s+socket\b",               "socket import"),
    (r"from\s+socket\b",                 "socket import"),
    (r"\bsocket\.socket\b",              "socket usage"),
    (r"import\s+urllib\b",               "urllib"),
    (r"from\s+urllib\b",                 "urllib"),
    (r"import\s+requests\b",             "requests"),
    (r"from\s+requests\b",               "requests"),
    (r"import\s+http\.client\b",         "http.client"),
    (r"import\s+httplib\b",              "httplib"),
    (r"\bos\.remove\s*\(",               "os.remove"),
    (r"\bos\.unlink\s*\(",               "os.unlink"),
    (r"\bos\.rmdir\s*\(",                "os.rmdir"),
    (r"\bos\.makedirs\s*\(",             "os.makedirs"),
    (r"\bos\.mkdir\s*\(",                "os.mkdir"),
    (r"\bshutil\b",                      "shutil"),
    (r"\bctypes\b",                      "ctypes"),
    (r"\bcffi\b",                        "cffi"),
    (r"\b__import__\s*\(",               "dynamic __import__"),
    (r"\bpickle\.loads?\s*\(",           "pickle.load"),
    # write-mode file opens; read-mode is allowed
    (r"open\s*\([^)]*,\s*['\"]w[ab+]?['\"]", "file write-open"),
    (r"open\s*\([^)]*,\s*['\"]a['\"]",        "file append-open"),
]

_COMPILED_BLOCKED = [(re.compile(pat), reason) for pat, reason in _BLOCKED]


@dataclass
class ScanResult:
    safe: bool
    reason: str = ""


def safety_scan(code: str) -> ScanResult:
    """Return ScanResult(safe=True) or ScanResult(safe=False, reason=...).

    Scans the source text with compiled regexes. Fast — no execution.
    """
    for regex, reason in _COMPILED_BLOCKED:
        if regex.search(code):
            return ScanResult(safe=False, reason=reason)
    return ScanResult(safe=True)


# ── Sandboxed execution ──────────────────────────────────────────────────────

_MEMORY_LIMIT_BYTES = 256 * 1024 * 1024   # 256 MB
_CPU_TIME_LIMIT_S   = 30                   # hard CPU-second cap per process
DEFAULT_TIMEOUT_S   = 10.0                 # wall-clock per test case
DEFAULT_MAX_CASES   = 5                    # cap so eval stays fast


@dataclass
class ExecResult:
    passed: bool
    n_passed: int
    n_total: int
    stderr: str = ""
    timed_out: bool = False
    scan_blocked: bool = False
    scan_reason: str = ""
    details: list[dict] = field(default_factory=list)


def _set_limits() -> None:
    """Called as preexec_fn inside the child process."""
    resource.setrlimit(resource.RLIMIT_AS,  (_MEMORY_LIMIT_BYTES, _MEMORY_LIMIT_BYTES))
    resource.setrlimit(resource.RLIMIT_CPU, (_CPU_TIME_LIMIT_S,   _CPU_TIME_LIMIT_S))


def _run_stdio(
    code: str,
    input_str: str,
    expected_output: str,
    timeout: float,
) -> dict:
    """Run code with stdin, compare normalised stdout to expected. Returns a detail dict."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            input=input_str,
            capture_output=True,
            text=True,
            timeout=timeout,
            preexec_fn=_set_limits,
        )
        actual = proc.stdout.strip()
        expected = str(expected_output).strip()
        passed = actual == expected
        return {
            "passed": passed,
            "actual": actual[:200],
            "expected": expected[:200],
            "stderr": proc.stderr[:200],
            "timed_out": False,
        }
    except subprocess.TimeoutExpired:
        return {"passed": False, "actual": "", "expected": "", "stderr": "", "timed_out": True}
    except Exception as exc:
        return {"passed": False, "actual": "", "expected": "", "stderr": str(exc), "timed_out": False}


def _run_fn_call(
    code: str,
    fn_name: str,
    inputs: list,
    outputs: list,
    timeout: float,
    max_cases: int,
) -> list[dict]:
    """Run function-call style tests by building a harness script."""
    pairs = list(zip(inputs, outputs))[:max_cases]
    harness = textwrap.dedent(f"""
{code}

import json as _json, sys as _sys
_pairs = _json.loads({json.dumps(json.dumps(pairs))})
_results = []
for _inp, _exp in _pairs:
    try:
        _args = _inp if isinstance(_inp, list) else [_inp]
        _out = {fn_name}(*_args)
        _pass = (_out == _exp)
        _results.append({{"passed": _pass, "actual": repr(_out)[:200], "expected": repr(_exp)[:200], "stderr": "", "timed_out": False}})
    except Exception as _e:
        _results.append({{"passed": False, "actual": "", "expected": repr(_exp)[:200], "stderr": str(_e)[:200], "timed_out": False}})
print(_json.dumps(_results))
""")
    try:
        proc = subprocess.run(
            [sys.executable, "-c", harness],
            capture_output=True,
            text=True,
            timeout=timeout * max_cases,
            preexec_fn=_set_limits,
        )
        return json.loads(proc.stdout.strip())
    except subprocess.TimeoutExpired:
        return [{"passed": False, "actual": "", "expected": "", "stderr": "", "timed_out": True}]
    except Exception as exc:
        return [{"passed": False, "actual": "", "expected": "", "stderr": str(exc), "timed_out": False}]


def execute_solution(
    code: str,
    input_output: dict,
    timeout: float = DEFAULT_TIMEOUT_S,
    max_cases: int = DEFAULT_MAX_CASES,
) -> ExecResult:
    """Scan then execute; return an ExecResult.

    input_output format:
        {"inputs": [...], "outputs": [...]}           — stdio
        {"fn_name": "f", "inputs": [...], "outputs": [...]} — function call
    """
    scan = safety_scan(code)
    if not scan.safe:
        return ExecResult(
            passed=False, n_passed=0, n_total=0,
            scan_blocked=True, scan_reason=scan.reason,
        )

    fn_name = input_output.get("fn_name")
    raw_inputs  = (input_output.get("inputs")  or [])[:max_cases]
    raw_outputs = (input_output.get("outputs") or [])[:max_cases]

    if not raw_inputs:
        return ExecResult(passed=False, n_passed=0, n_total=0, stderr="no test cases")

    details: list[dict] = []
    any_timeout = False

    if fn_name:
        details = _run_fn_call(code, fn_name, raw_inputs, raw_outputs, timeout, max_cases)
        any_timeout = any(d.get("timed_out") for d in details)
    else:
        for inp, exp in zip(raw_inputs, raw_outputs):
            d = _run_stdio(code, str(inp), str(exp), timeout)
            details.append(d)
            if d["timed_out"]:
                any_timeout = True
                break   # stop running if first case times out

    n_passed = sum(1 for d in details if d["passed"])
    n_total  = len(details)
    first_err = next((d["stderr"] for d in details if d.get("stderr")), "")

    return ExecResult(
        passed=n_passed == n_total and n_total > 0,
        n_passed=n_passed,
        n_total=n_total,
        stderr=first_err,
        timed_out=any_timeout,
        details=details,
    )


# ── CLI smoke test ────────────────────────────────────────────────────────────

def main() -> None:
    """Quick smoke test: scan and run a trivial solution."""
    code = "n = int(input()); print(n * n)"
    io = {"inputs": ["5", "3"], "outputs": ["25", "9"]}
    scan = safety_scan(code)
    print(f"scan: {scan}")
    result = execute_solution(code, io)
    print(f"result: passed={result.passed} n_passed={result.n_passed}/{result.n_total}")

    bad_code = "import subprocess; subprocess.run(['ls'])"
    scan2 = safety_scan(bad_code)
    print(f"bad scan: {scan2}")


if __name__ == "__main__":
    main()
