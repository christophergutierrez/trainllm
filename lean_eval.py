#!/usr/bin/env python3
"""
Lean tactic evaluation runner.

Reads test.jsonl, generates one tactic per record using an MLX model,
safety-checks each tactic, optionally verifies with Lean, and writes
per-record prediction JSONL + compiler_errors.jsonl + run_config.json +
summary JSON.  Output field names match the spec in FULL_EVAL.md.

Requires mlx_lm on Mac (minimum version 0.21.0):
  pip install "mlx-lm>=0.21.0"

Usage:
  # Target-only (no speculative decoding)
  python3 scripts/lean_eval.py --model fused-7b-lean \\
      --test data/lean_stat/test.jsonl \\
      --output reports/lean_eval/target-7b

  # Speculative decoding
  python3 scripts/lean_eval.py --model fused-7b-lean \\
      --draft-model fused-0.5b-lean \\
      --num-draft-tokens 5 \\
      --test data/lean_stat/test.jsonl \\
      --output reports/lean_eval/speculative
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

MLX_LM_MIN_VERSION = "0.21.0"

SYSTEM_PROMPT = (
    "You are a Lean 4 proof assistant. "
    "Given a tactic proof state, provide exactly one tactic that makes progress."
)

CHAT_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{user}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def _check_mlx_version() -> None:
    try:
        import importlib.metadata
        ver = importlib.metadata.version("mlx-lm")
        from packaging.version import Version
        if Version(ver) < Version(MLX_LM_MIN_VERSION):
            print(f"WARNING: mlx-lm {ver} < required {MLX_LM_MIN_VERSION}. "
                  f"Upgrade: pip install 'mlx-lm>={MLX_LM_MIN_VERSION}'",
                  file=sys.stderr)
    except Exception:
        pass


def _build_prompt(state_before: str) -> str:
    user = f"Given the Lean 4 state:\n{state_before}\nProvide the next tactical step."
    return CHAT_TEMPLATE.format(system=SYSTEM_PROMPT, user=user)


def _load_mlx_model(model_path: str, draft_model_path: str | None):
    try:
        from mlx_lm import load
    except ImportError:
        sys.exit("mlx_lm not found. Install: pip install 'mlx-lm>=0.21.0'")
    model, tokenizer = load(model_path)
    draft = None
    if draft_model_path:
        draft, _ = load(draft_model_path)
    return model, tokenizer, draft


def _generate(model, tokenizer, draft, prompt: str, max_tokens: int,
              num_draft_tokens: int, temp: float) -> tuple[str, float]:
    from mlx_lm import generate
    t0 = time.monotonic()
    kwargs: dict = dict(max_tokens=max_tokens, temp=temp)
    # verbose kwarg added in mlx-lm 0.19; pass only if accepted
    try:
        import inspect
        if "verbose" in inspect.signature(generate).parameters:
            kwargs["verbose"] = False
    except Exception:
        pass
    if draft is not None:
        kwargs["draft_model"] = draft
        kwargs["num_draft_tokens"] = num_draft_tokens
    response = generate(model, tokenizer, prompt=prompt, **kwargs)
    elapsed = time.monotonic() - t0
    response = response.split("<|im_end|>")[0].strip()
    return response, elapsed


def _count_tokens(text: str, tokenizer) -> int:
    ids = tokenizer.encode(text)
    return len(ids) if isinstance(ids, list) else ids.shape[-1]


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return "unknown"


def run_eval(
    model_path: str,
    test_path: Path,
    out_dir: Path,
    draft_model_path: str | None = None,
    num_draft_tokens: int = 5,
    max_tokens: int = 256,
    temp: float = 0.0,
    limit: int | None = None,
    lean_timeout: int = 60,
    skip_lean: bool = False,
) -> dict:
    from lean_verify import safety_check, verify_tactic

    _check_mlx_version()
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    with open(test_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if limit is not None:
        records = records[:limit]

    # Write run config
    run_cfg = {
        "model": model_path,
        "draft_model": draft_model_path,
        "num_draft_tokens": num_draft_tokens if draft_model_path else None,
        "test_path": str(test_path),
        "max_tokens": max_tokens,
        "temp": temp,
        "limit": limit,
        "skip_lean": skip_lean,
        "git_commit": _git_commit(),
        "platform": platform.platform(),
        "python": sys.version,
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2))

    model, tokenizer, draft = _load_mlx_model(model_path, draft_model_path)

    pred_path = out_dir / "predictions.jsonl"
    err_path = out_dir / "compiler_errors.jsonl"

    n_safety_fail = 0
    n_lean_pass = 0
    n_lean_fail = 0
    n_lean_skip = 0
    total_tokens = 0
    total_elapsed = 0.0

    with open(pred_path, "w") as pred_f, open(err_path, "w") as err_f:
        for i, rec in enumerate(records):
            convs = rec.get("conversations", [])
            if convs:
                state_before = expected_tactic = ""
                for c in convs:
                    if c.get("from") == "human":
                        val = c.get("value", "")
                        import re
                        m = re.search(
                            r"Given the Lean 4 state:\n(.*?)\nProvide", val, re.S)
                        state_before = m.group(1).strip() if m else val
                    elif c.get("from") == "gpt":
                        expected_tactic = c.get("value", "").strip()
            else:
                state_before = rec.get("state_before", "")
                expected_tactic = rec.get("tactic", "")

            prompt = _build_prompt(state_before)
            gen_text, elapsed = _generate(model, tokenizer, draft, prompt,
                                          max_tokens=max_tokens,
                                          num_draft_tokens=num_draft_tokens,
                                          temp=temp)
            n_tokens = _count_tokens(gen_text, tokenizer)
            tps = n_tokens / elapsed if elapsed > 0 else 0.0
            total_tokens += n_tokens
            total_elapsed += elapsed

            forbidden = not safety_check(gen_text)
            if forbidden:
                n_safety_fail += 1

            lean_pass = None
            lean_stdout = lean_stderr = ""
            if not forbidden and not skip_lean:
                vr = verify_tactic(state_before, gen_text, timeout=lean_timeout)
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
                "prompt": prompt,
                "state_before": state_before,
                "expected_tactic": expected_tactic,
                "generated_text": gen_text,
                "generated_tactic": gen_text,
                "forbidden_token": forbidden,
                "lean_pass": lean_pass,
                "lean_stdout": lean_stdout,
                "lean_stderr": lean_stderr,
                "elapsed_seconds": round(elapsed, 3),
                "generated_tokens": n_tokens,
                "tokens_per_second": round(tps, 1),
            }
            pred_f.write(json.dumps(row, ensure_ascii=False) + "\n")

            if lean_pass is False and lean_stderr:
                err_f.write(json.dumps({
                    "index": i,
                    "generated_tactic": gen_text,
                    "lean_stderr": lean_stderr,
                }, ensure_ascii=False) + "\n")

            print(f"[{i+1}/{len(records)}] forbidden={forbidden} "
                  f"lean={lean_pass} tps={tps:.1f}", flush=True)

    n_total = len(records)
    lean_eligible = n_lean_pass + n_lean_fail
    summary = {
        "model": model_path,
        "draft_model": draft_model_path,
        "num_draft_tokens": num_draft_tokens if draft_model_path else None,
        "test_path": str(test_path),
        "n_total": n_total,
        "n_safety_fail": n_safety_fail,
        "n_lean_pass": n_lean_pass,
        "n_lean_fail": n_lean_fail,
        "n_lean_skip": n_lean_skip,
        "compile_pass_rate": n_lean_pass / lean_eligible if lean_eligible > 0 else None,
        "mean_elapsed_s": round(total_elapsed / n_total, 3) if n_total else 0,
        "mean_tps": round(total_tokens / total_elapsed, 1) if total_elapsed > 0 else 0,
        "total_tokens": total_tokens,
        "total_elapsed_s": round(total_elapsed, 1),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True,
                   help="Path to fused MLX model directory")
    p.add_argument("--draft-model", default=None,
                   help="Path to fused draft MLX model (enables speculative decoding)")
    p.add_argument("--num-draft-tokens", type=int, default=5)
    p.add_argument("--test", type=Path, default=Path("data/lean_stat/test.jsonl"),
                   help="Test JSONL file (default: data/lean_stat/test.jsonl)")
    p.add_argument("--output", type=Path,
                   default=Path("reports/lean_eval/run"),
                   help="Output directory for predictions.jsonl, summary.json, etc.")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temp", type=float, default=0.0,
                   help="Sampling temperature (0 = greedy)")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap number of test records (useful for smoke tests)")
    p.add_argument("--lean-timeout", type=int, default=60)
    p.add_argument("--skip-lean", action="store_true",
                   help="Skip Lean compilation (safety check only)")
    args = p.parse_args()

    summary = run_eval(
        model_path=args.model,
        test_path=args.test,
        out_dir=args.output,
        draft_model_path=args.draft_model,
        num_draft_tokens=args.num_draft_tokens,
        max_tokens=args.max_tokens,
        temp=args.temp,
        limit=args.limit,
        lean_timeout=args.lean_timeout,
        skip_lean=args.skip_lean,
    )
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
