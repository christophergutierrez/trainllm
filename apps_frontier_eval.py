#!/usr/bin/env python3
"""
Frontier model reference evaluation for the APPS benchmark.

Default mode: in-session (uses the Claude Code session model via subagents —
run this from a Workflow, not directly). Add --use-api to call the Anthropic
API instead; this incurs token charges.

Hard limit: 10 problems. This is enough to confirm the pipeline works. Pass
--allow-more to lift the cap (only do this intentionally).

Usage:
    # In-session default (10 problems, no charge)
    python3 apps_frontier_eval.py \\
        --test data/apps/eval_all.jsonl \\
        --output results/apps/frontier-no-context

    # Paid API (10 problems)
    python3 apps_frontier_eval.py --use-api \\
        --test data/apps/eval_all.jsonl \\
        --output results/apps/frontier-no-context

    # With context (paid API, 10 problems)
    python3 apps_frontier_eval.py --use-api --mode with-context \\
        --train data/apps/train.jsonl \\
        --test data/apps/eval_all.jsonl \\
        --output results/apps/frontier-with-context
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

MODEL = "claude-opus-4-8"
DEFAULT_N_SHOTS = 3
MAX_OUTPUT_TOKENS = 1024
HARD_LIMIT = 10

SYSTEM_PROMPT = (
    "You are a Python programming assistant. "
    "Write a complete, correct Python solution for the given programming problem. "
    "Output only the code — no explanation, no markdown fences."
)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return "unknown"


def _load_records(path: Path, limit: int | None) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def _build_messages(question: str, starter_code: str, shots: list[dict]) -> list[dict]:
    msgs: list[dict] = []
    for shot in shots:
        user_text = shot["question"].strip()
        if shot.get("starter_code"):
            user_text += f"\n\nStarter code:\n{shot['starter_code'].strip()}"
        msgs.append({"role": "user", "content": user_text})
        msgs.append({"role": "assistant", "content": shot["solution"].strip()})
    user_text = question.strip()
    if starter_code and starter_code.strip():
        user_text += f"\n\nStarter code:\n{starter_code.strip()}"
    msgs.append({"role": "user", "content": user_text})
    return msgs


def run_eval(
    test_path: Path,
    out_dir: Path,
    mode: str = "no-context",
    train_path: Path | None = None,
    n_shots: int = DEFAULT_N_SHOTS,
    limit: int = HARD_LIMIT,
    exec_timeout: float = 10.0,
    max_cases: int = 5,
    skip_verify: bool = False,
    use_api: bool = False,
    allow_more: bool = False,
    seed: int = 42,
) -> dict:

    if not use_api:
        sys.exit(
            "This script makes paid API calls without --use-api.\n"
            "For in-session evaluation (no charge), run via a Workflow instead.\n"
            "Add --use-api only if you intentionally want billable API usage."
        )

    if limit > HARD_LIMIT and not allow_more:
        sys.exit(
            f"Limit {limit} exceeds the safety cap of {HARD_LIMIT}.\n"
            f"Pass --allow-more to confirm you want to run more than {HARD_LIMIT} problems."
        )

    try:
        import anthropic
    except ImportError:
        sys.exit("anthropic package not found. Install: pip install anthropic")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ANTHROPIC_API_KEY not set in environment")

    client = anthropic.Anthropic(api_key=api_key)

    sys.path.insert(0, str(Path(__file__).parent))
    from apps_verify import safety_scan, execute_solution

    out_dir.mkdir(parents=True, exist_ok=True)
    records = _load_records(test_path, limit)

    shots: list[dict] = []
    if mode == "with-context" and train_path is not None and train_path.exists():
        from apps_context import load_train_examples
        train_examples = load_train_examples(train_path)
        import random
        rng = random.Random(seed)
        short = [e for e in train_examples if len(e.question) < 1500 and e.solution]
        sample = rng.sample(short, min(n_shots, len(short)))
        shots = [{"question": e.question, "starter_code": "", "solution": e.solution}
                 for e in sample]

    run_cfg = {
        "model": MODEL,
        "mode": mode,
        "n_shots": len(shots),
        "limit": limit,
        "use_api": True,
        "api_provider": "anthropic",
        "test_path": str(test_path),
        "train_path": str(train_path) if train_path else None,
        "skip_verify": skip_verify,
        "exec_timeout": exec_timeout,
        "max_cases": max_cases,
        "git_commit": _git_commit(),
        "platform": platform.platform(),
        "python": sys.version,
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2))
    (out_dir / "summary.json").unlink(missing_ok=True)

    pred_path = out_dir / "predictions.jsonl"
    fail_path = out_dir / "failures.jsonl"

    n_pass = n_fail = n_skip = n_scan_blocked = 0
    total_input_toks = total_output_toks = 0
    total_elapsed = 0.0

    import re

    def _strip_fences(text: str) -> str:
        text = re.sub(r"^```(?:python)?\s*\n?", "", text.strip())
        text = re.sub(r"\n?```\s*$", "", text).strip()
        return text

    with pred_path.open("w") as pf, fail_path.open("w") as ff:
        for i, rec in enumerate(records):
            question   = rec.get("question", "")
            starter    = rec.get("starter_code", "")
            io         = rec.get("input_output", {})
            difficulty = rec.get("difficulty", "")

            messages = _build_messages(question, starter, shots)

            t0 = time.monotonic()
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_OUTPUT_TOKENS,
                system=SYSTEM_PROMPT,
                messages=messages,
            )
            elapsed = time.monotonic() - t0
            code = _strip_fences(response.content[0].text)
            in_toks  = response.usage.input_tokens
            out_toks = response.usage.output_tokens
            total_input_toks  += in_toks
            total_output_toks += out_toks
            total_elapsed += elapsed
            tps = out_toks / elapsed if elapsed > 0 else 0.0

            passed = scan_blocked = False
            scan_reason = stderr = ""
            n_cases_passed = n_cases_total = 0
            timed_out = False

            if skip_verify or not io:
                n_skip += 1
            else:
                result = execute_solution(code, io, timeout=exec_timeout, max_cases=max_cases)
                passed       = result.passed
                scan_blocked = result.scan_blocked
                scan_reason  = result.scan_reason
                stderr       = result.stderr
                timed_out    = result.timed_out
                n_cases_passed = result.n_passed
                n_cases_total  = result.n_total

                if scan_blocked:
                    n_scan_blocked += 1
                    n_skip += 1
                elif passed:
                    n_pass += 1
                else:
                    n_fail += 1

            row = {
                "index": i,
                "problem_id": rec.get("problem_id"),
                "difficulty": difficulty,
                "generated_code": code,
                "passed": passed,
                "scan_blocked": scan_blocked,
                "scan_reason": scan_reason,
                "n_cases_passed": n_cases_passed,
                "n_cases_total": n_cases_total,
                "timed_out": timed_out,
                "stderr": stderr[:200],
                "elapsed_s": round(elapsed, 3),
                "input_tokens": in_toks,
                "output_tokens": out_toks,
                "tps": round(tps, 1),
            }
            pf.write(json.dumps(row, ensure_ascii=False) + "\n")
            if not passed and not scan_blocked and io:
                rec_out = dict(rec)
                rec_out.update({"generated_code": code, "stderr": stderr[:400]})
                ff.write(json.dumps(rec_out, ensure_ascii=False) + "\n")

            eligible = n_pass + n_fail
            rate_str = f"{n_pass/eligible*100:.1f}%" if eligible > 0 else "?"
            print(
                f"[{i+1}/{len(records)}] {difficulty} pass={rate_str} "
                f"in={in_toks} out={out_toks} tps={tps:.1f}",
                flush=True,
            )

    n_total = len(records)
    eligible = n_pass + n_fail
    n_evaluated = n_total - n_skip
    summary = {
        "model": MODEL,
        "mode": mode,
        "n_shots": len(shots),
        "n_total": n_total,
        "n_evaluated": n_evaluated,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_skip": n_skip,
        "n_scan_blocked": n_scan_blocked,
        "pass_rate": n_pass / eligible if eligible > 0 else None,
        "mean_elapsed_s": round(total_elapsed / n_total, 3) if n_total else 0,
        "mean_tps": round(total_output_toks / total_elapsed, 1) if total_elapsed > 0 else 0,
        "total_input_tokens": total_input_toks,
        "total_output_tokens": total_output_toks,
        "mean_input_tokens": round(total_input_toks / n_total, 1) if n_total else 0,
        "mean_output_tokens": round(total_output_toks / n_total, 1) if n_total else 0,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mode", choices=["no-context", "with-context"], default="no-context")
    p.add_argument("--use-api", action="store_true",
                   help="Make paid Anthropic API calls (required)")
    p.add_argument("--allow-more", action="store_true",
                   help=f"Lift the {HARD_LIMIT}-problem safety cap")
    p.add_argument("--test", type=Path, default=Path("data/apps/eval_all.jsonl"))
    p.add_argument("--train", type=Path, default=Path("data/apps/train.jsonl"))
    p.add_argument("--n-shots", type=int, default=DEFAULT_N_SHOTS)
    p.add_argument("--output", type=Path, default=Path("results/apps/frontier-no-context"))
    p.add_argument("--limit", type=int, default=HARD_LIMIT,
                   help=f"Max problems (default {HARD_LIMIT}; hard cap unless --allow-more)")
    p.add_argument("--skip-verify", action="store_true")
    p.add_argument("--exec-timeout", type=float, default=10.0)
    p.add_argument("--max-cases", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    summary = run_eval(
        test_path=args.test,
        out_dir=args.output,
        mode=args.mode,
        train_path=args.train,
        n_shots=args.n_shots,
        limit=args.limit,
        exec_timeout=args.exec_timeout,
        max_cases=args.max_cases,
        skip_verify=args.skip_verify,
        use_api=args.use_api,
        allow_more=args.allow_more,
        seed=args.seed,
    )
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
