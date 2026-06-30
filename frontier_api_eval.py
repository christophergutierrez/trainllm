#!/usr/bin/env python3
"""
Paid frontier API baseline for the Lean tactic prediction task.

Two modes:
  no-context    Zero-shot: model sees only the tactic state.
  with-context  Few-shot: model sees N training examples before each query.

These runs answer whether fine-tuning was necessary versus prompting a
frontier model through a billable API, and at what token cost.  Token counts
(input + output) are recorded per row so the report can compare cost vs
quality.

Output format matches lean_eval.py (predictions.jsonl, summary.json,
run_config.json, compiler_errors.jsonl).  Two additional fields are added
per prediction: input_tokens and output_tokens.

Usage:
  python3 scripts/frontier_api_eval.py --use-api \\
      --mode no-context \\
      --test data/lean_stat/test.jsonl \\
      --output reports/lean_eval/frontier-api-no-context

  python3 scripts/frontier_api_eval.py --use-api \\
      --mode with-context \\
      --train data/lean_stat/train.jsonl \\
      --n-shots 5 \\
      --test data/lean_stat/test.jsonl \\
      --output reports/lean_eval/frontier-api-with-context

This script makes paid API calls. It refuses to run unless --use-api is passed.
Requires ANTHROPIC_API_KEY in the environment after confirmation.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import re
import subprocess
import sys
import time
from pathlib import Path

MODEL = "claude-opus-4-8"
DEFAULT_N_SHOTS = 5
MAX_OUTPUT_TOKENS = 256

SYSTEM_PROMPT = (
    "You are a Lean 4 proof assistant. "
    "Given a tactic proof state, provide exactly one tactic that makes progress. "
    "Output only the tactic itself — no explanation, no code fences, no markdown."
)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return "unknown"


def _load_records(path: Path, limit: int | None = None) -> list[dict]:
    records: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            convs = rec.get("conversations", [])
            if convs:
                state_before = expected = ""
                for c in convs:
                    if c.get("from") == "human":
                        m = re.search(
                            r"Given the Lean 4 state:\n(.*?)\nProvide",
                            c.get("value", ""), re.S)
                        state_before = m.group(1).strip() if m else c.get("value", "")
                    elif c.get("from") == "gpt":
                        expected = c.get("value", "").strip()
                rec = {"state_before": state_before, "expected_tactic": expected}
            records.append(rec)
            if limit is not None and len(records) >= limit:
                break
    return records


def _strip_markdown(text: str) -> str:
    text = re.sub(r"^```(?:lean4?|lean)?\s*\n?", "", text.strip())
    text = re.sub(r"\n?```\s*$", "", text).strip()
    # If the model emits multiple lines (explanation + tactic), take the last
    # non-empty line that looks like a tactic (starts with a lowercase keyword)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[-1] if lines else text


def _build_no_context(state_before: str) -> list[dict]:
    return [{
        "role": "user",
        "content": (
            f"Given the Lean 4 state:\n{state_before}\n"
            "Provide the next tactical step."
        ),
    }]


def _build_with_context(state_before: str, shots: list[dict]) -> list[dict]:
    messages: list[dict] = []
    for shot in shots:
        messages.append({
            "role": "user",
            "content": (
                f"Given the Lean 4 state:\n{shot['state_before']}\n"
                "Provide the next tactical step."
            ),
        })
        messages.append({"role": "assistant", "content": shot["expected_tactic"]})
    messages.append({
        "role": "user",
        "content": (
            f"Given the Lean 4 state:\n{state_before}\n"
            "Provide the next tactical step."
        ),
    })
    return messages


def run_eval(
    test_path: Path,
    out_dir: Path,
    mode: str,
    train_path: Path | None = None,
    n_shots: int = DEFAULT_N_SHOTS,
    limit: int | None = None,
    lean_timeout: int = 60,
    skip_lean: bool = False,
    seed: int = 42,
    use_api: bool = False,
    lean_project: Path | None = None,
    max_state_chars: int | None = None,
) -> dict:
    if not use_api:
        sys.exit(
            "Refusing to make paid frontier API calls without explicit confirmation.\n"
            "This path is separate from ChatGPT/Claude subscription plans and may "
            "incur token-based API charges.\n"
            "Re-run with --use-api only if you intentionally want billable API usage."
        )

    try:
        import anthropic
    except ImportError:
        sys.exit("anthropic package not found. Install: pip install anthropic")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ANTHROPIC_API_KEY not set in environment")

    client = anthropic.Anthropic(api_key=api_key)

    # lean_verify may live beside this script (bundle) or at repo root
    sys.path.insert(0, str(Path(__file__).parent))
    from lean_verify import safety_check, verify_tactic

    out_dir.mkdir(parents=True, exist_ok=True)
    records = _load_records(test_path, limit=limit)

    shots: list[dict] = []
    if mode == "with-context":
        if train_path is None or not train_path.exists():
            sys.exit("--train is required for --mode with-context and the file must exist")
        all_train = _load_records(train_path)
        rng = random.Random(seed)
        # Pick short records so the few-shot block doesn't balloon the prompt
        short = [r for r in all_train
                 if len(r.get("state_before", "")) < 400
                 and r.get("expected_tactic", "")]
        shots = rng.sample(short, min(n_shots, len(short)))

    run_cfg = {
        "model": MODEL,
        "mode": mode,
        "n_shots": len(shots),
        "test_path": str(test_path),
        "train_path": str(train_path) if train_path else None,
        "limit": limit,
        "skip_lean": skip_lean,
        "lean_project": str(lean_project) if lean_project else None,
        "max_state_chars": max_state_chars,
        "seed": seed,
        "api_confirmed": True,
        "api_provider": "anthropic",
        "git_commit": _git_commit(),
        "platform": platform.platform(),
        "python": sys.version,
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2))

    pred_path = out_dir / "predictions.jsonl"
    err_path = out_dir / "compiler_errors.jsonl"
    summary_path = out_dir / "summary.json"
    summary_path.unlink(missing_ok=True)

    n_safety_fail = 0
    n_lean_pass = 0
    n_lean_fail = 0
    n_lean_skip = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_elapsed = 0.0
    n_skipped_size = 0
    n_evaluated = 0

    with open(pred_path, "w") as pred_f, open(err_path, "w") as err_f:
        for i, rec in enumerate(records):
            state_before = rec.get("state_before", "")
            expected_tactic = rec.get("expected_tactic", "")

            if max_state_chars is not None and len(state_before) > max_state_chars:
                n_skipped_size += 1
                continue

            if mode == "no-context":
                messages = _build_no_context(state_before)
            else:
                messages = _build_with_context(state_before, shots)

            t0 = time.monotonic()
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_OUTPUT_TOKENS,
                system=SYSTEM_PROMPT,
                messages=messages,
            )
            elapsed = time.monotonic() - t0
            n_evaluated += 1

            gen_text = _strip_markdown(response.content[0].text)
            in_toks = response.usage.input_tokens
            out_toks = response.usage.output_tokens
            total_input_tokens += in_toks
            total_output_tokens += out_toks
            total_elapsed += elapsed
            tps = out_toks / elapsed if elapsed > 0 else 0.0

            forbidden = not safety_check(gen_text)
            if forbidden:
                n_safety_fail += 1

            lean_pass = None
            lean_stdout = lean_stderr = ""
            if not forbidden and not skip_lean:
                vr = verify_tactic(state_before, gen_text, timeout=lean_timeout,
                                   project_dir=lean_project)
                lean_pass = vr.lean_ok
                lean_stdout = vr.stdout
                lean_stderr = vr.stderr
                if lean_pass is None:
                    n_lean_skip += 1
                elif lean_pass:
                    n_lean_pass += 1
                else:
                    n_lean_fail += 1
            else:
                n_lean_skip += 1

            row = {
                "index": i,
                "prompt": json.dumps(messages),
                "state_before": state_before,
                "expected_tactic": expected_tactic,
                "generated_text": gen_text,
                "generated_tactic": gen_text,
                "forbidden_token": forbidden,
                "lean_pass": lean_pass,
                "lean_stdout": lean_stdout,
                "lean_stderr": lean_stderr,
                "elapsed_seconds": round(elapsed, 3),
                "generated_tokens": out_toks,
                "tokens_per_second": round(tps, 1),
                "input_tokens": in_toks,
                "output_tokens": out_toks,
            }
            pred_f.write(json.dumps(row, ensure_ascii=False) + "\n")

            if lean_pass is False and (lean_stderr or lean_stdout):
                err_f.write(json.dumps({
                    "index": i,
                    "generated_tactic": gen_text,
                    "lean_stdout": lean_stdout,
                    "lean_stderr": lean_stderr,
                }, ensure_ascii=False) + "\n")

            print(
                f"[{i+1}/{len(records)}] in={in_toks} out={out_toks} "
                f"forbidden={forbidden} lean={lean_pass} tps={tps:.1f}",
                flush=True,
            )

    n_total = len(records)
    lean_eligible = n_lean_pass + n_lean_fail
    summary = {
        "model": MODEL,
        "mode": mode,
        "n_shots": len(shots),
        "test_path": str(test_path),
        "n_total": n_total,
        "n_skipped_size": n_skipped_size,
        "n_evaluated": n_evaluated,
        "n_safety_fail": n_safety_fail,
        "n_lean_pass": n_lean_pass,
        "n_lean_fail": n_lean_fail,
        "n_lean_skip": n_lean_skip,
        "compile_pass_rate": n_lean_pass / lean_eligible if lean_eligible > 0 else None,
        "mean_elapsed_s": round(total_elapsed / n_evaluated, 3) if n_evaluated else 0,
        "mean_tps": round(total_output_tokens / total_elapsed, 1) if total_elapsed > 0 else 0,
        "total_tokens": total_output_tokens,
        "total_elapsed_s": round(total_elapsed, 1),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "mean_input_tokens": round(total_input_tokens / n_evaluated, 1) if n_evaluated else 0,
        "mean_output_tokens": round(total_output_tokens / n_evaluated, 1) if n_evaluated else 0,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mode", choices=["no-context", "with-context"], required=True,
                   help="no-context: zero-shot; with-context: few-shot from training data")
    p.add_argument("--use-api", action="store_true",
                   help="Required confirmation: make billable frontier API calls")
    p.add_argument("--test", type=Path, default=Path("data/lean_stat/test.jsonl"))
    p.add_argument("--train", type=Path, default=Path("data/lean_stat/train.jsonl"),
                   help="Training data for few-shot shots (with-context mode only)")
    p.add_argument("--n-shots", type=int, default=DEFAULT_N_SHOTS,
                   help=f"Number of few-shot examples (default: {DEFAULT_N_SHOTS})")
    p.add_argument("--output", type=Path, default=Path("reports/lean_eval/frontier-api"),
                   help="Output directory for predictions.jsonl, summary.json, etc.")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap number of test records (useful for smoke tests)")
    p.add_argument("--lean-timeout", type=int, default=60)
    p.add_argument("--skip-lean", action="store_true",
                   help="Skip Lean compilation (safety check only)")
    p.add_argument("--lean-project", type=Path, default=None,
                   help="Lake project root for `lake env lean` verification "
                        "(e.g. ~/git_home/lean-stat-learning-theory)")
    p.add_argument("--max-state-chars", type=int, default=None,
                   help="Skip records where state exceeds this many chars")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for few-shot sampling")
    args = p.parse_args()

    summary = run_eval(
        test_path=args.test,
        out_dir=args.output,
        mode=args.mode,
        train_path=args.train,
        n_shots=args.n_shots,
        limit=args.limit,
        lean_timeout=args.lean_timeout,
        skip_lean=args.skip_lean,
        seed=args.seed,
        use_api=args.use_api,
        lean_project=args.lean_project,
        max_state_chars=args.max_state_chars,
    )
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
