#!/usr/bin/env python3
"""
APPS coding benchmark evaluation runner.

Reads an APPS eval JSONL (from prepare_apps_data.py), generates one Python
solution per problem, safety-scans and executes it against the bundled test
cases, and writes predictions + summary.

Output (per run directory):
    predictions.jsonl     one record per problem
    summary.json          aggregate metrics
    run_config.json       full reproducibility record
    failures.jsonl        problems where all test cases failed (for debugging)

Backends:
    mlx    Apple MLX — default on Mac, requires mlx-lm>=0.21.0
    hf     HuggingFace/Unsloth — for GB10
    mock   Returns a fixed solution; use to validate pipeline without a model

Usage:
    # Target-only (Mac)
    python3 apps_eval.py --model fused-7b-apps \\
        --test data/apps/eval_all.jsonl \\
        --output results/apps/target-7b

    # Speculative decoding
    python3 apps_eval.py --model fused-7b-apps \\
        --draft-model fused-0.5b-apps --num-draft-tokens 5 \\
        --test data/apps/eval_all.jsonl \\
        --output results/apps/speculative

    # With retrieved context
    python3 apps_eval.py --model fused-7b-apps \\
        --mode with-context --train data/apps/train.jsonl \\
        --test data/apps/eval_all.jsonl \\
        --output results/apps/target-7b-context

    # GB10 (HF backend)
    python3 apps_eval.py --backend hf \\
        --model adapters/7b-apps/final \\
        --test data/apps/eval_all.jsonl \\
        --output results/apps/hf-target-7b

    # Smoke test (no model)
    python3 apps_eval.py --backend mock --model unused \\
        --test data/apps/eval_all.jsonl --limit 10 \\
        --output results/apps/smoke
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

MLX_LM_MIN_VERSION = "0.21.0"

SYSTEM_PROMPT = (
    "You are a Python programming assistant. "
    "Write a complete, correct Python solution for the given programming problem. "
    "Output only the code — no explanation, no markdown fences."
)

_MOCK_SOLUTIONS = [
    "n = int(input())\nprint(n * (n + 1) // 2)",
    "import sys\ndata = sys.stdin.read().split()\nprint(sum(int(x) for x in data))",
    "print(input())",
    "a, b = map(int, input().split())\nprint(a + b)",
    "n = int(input())\nprint(n * n)",
]

MAX_OUTPUT_TOKENS = 1024


# ── Backend factories ────────────────────────────────────────────────────────

def _make_mock_backend():
    counter = [0]

    def generate_fn(prompt: str, max_tokens: int, num_draft_tokens: int, temp: float):
        sol = _MOCK_SOLUTIONS[counter[0] % len(_MOCK_SOLUTIONS)]
        counter[0] += 1
        return sol, 0.001

    def count_tokens_fn(text: str) -> int:
        return len(text.split())

    return generate_fn, count_tokens_fn


def _make_mlx_backend(model_path: str, draft_model_path: str | None):
    try:
        import importlib.metadata
        from packaging.version import Version
        ver = importlib.metadata.version("mlx-lm")
        if Version(ver) < Version(MLX_LM_MIN_VERSION):
            print(f"WARNING: mlx-lm {ver} < {MLX_LM_MIN_VERSION}", file=sys.stderr)
    except Exception:
        pass

    try:
        from mlx_lm import load, generate as mlx_generate
    except ImportError:
        sys.exit("mlx_lm not found. Install: pip install 'mlx-lm>=0.21.0'\n"
                 "On GB10, use --backend hf instead.")

    model, tokenizer = load(model_path)
    draft = load(draft_model_path)[0] if draft_model_path else None

    import inspect
    _verbose_ok = "verbose" in inspect.signature(mlx_generate).parameters

    def generate_fn(prompt: str, max_tokens: int, num_draft_tokens: int, temp: float):
        t0 = time.monotonic()
        kwargs: dict = dict(max_tokens=max_tokens, temp=temp)
        if _verbose_ok:
            kwargs["verbose"] = False
        if draft is not None:
            kwargs["draft_model"] = draft
            kwargs["num_draft_tokens"] = num_draft_tokens
        resp = mlx_generate(model, tokenizer, prompt=prompt, **kwargs)
        elapsed = time.monotonic() - t0
        return resp.split("<|im_end|>")[0].strip(), elapsed

    def count_tokens_fn(text: str) -> int:
        ids = tokenizer.encode(text)
        return len(ids) if isinstance(ids, list) else ids.shape[-1]

    return generate_fn, count_tokens_fn


def _make_hf_backend(model_path: str, draft_model_path: str | None):
    import os
    import shutil
    import tempfile

    hf_home = os.environ.get("HF_HOME", str(Path.home() / "llm" / "models" / "hf"))
    os.environ.setdefault("HF_HOME", hf_home)

    def _resolve(path: str) -> str:
        # Just expand the path; HF_HOME is set so AutoTokenizer/AutoModel will
        # find the base model in the cache automatically from the model name.
        return str(Path(path).expanduser())

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        from peft import PeftModel
    except ImportError:
        sys.exit("transformers/peft/torch not found. Use --backend mlx on Mac.")

    resolved = _resolve(model_path)
    device = "cuda:0" if __import__("torch").cuda.is_available() else "cpu"
    # Force all layers onto a single GPU; "auto" can spill to CPU on some driver versions.
    device_map = {"": 0} if device == "cuda:0" else "cpu"

    cfg_path = Path(resolved) / "adapter_config.json"
    if cfg_path.exists():
        base_cfg = json.loads(cfg_path.read_text())
        base_name = base_cfg["base_model_name_or_path"]
        tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            base_name, dtype=torch.float16,
            device_map=device_map, trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, resolved)
        model = model.merge_and_unload()
    else:
        tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            resolved, dtype=torch.float16,
            device_map=device_map, trust_remote_code=True,
        )

    model.eval()

    def generate_fn(prompt: str, max_tokens: int, num_draft_tokens: int, temp: float):
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        t0 = time.monotonic()
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_tokens,
                do_sample=(temp > 0),
                temperature=temp if temp > 0 else 1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.monotonic() - t0
        text = tokenizer.decode(out[0, enc["input_ids"].shape[-1]:], skip_special_tokens=True)
        return text.strip(), elapsed

    def count_tokens_fn(text: str) -> int:
        return len(tokenizer.encode(text))

    return generate_fn, count_tokens_fn


# ── Prompt helpers ───────────────────────────────────────────────────────────

def _build_prompt(question: str, starter_code: str = "",
                  context_examples: list | None = None) -> str:
    parts = [f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"]
    if context_examples:
        from apps_context import format_context_block
        ctx = format_context_block(context_examples)
        parts.append(f"<|im_start|>user\nHere are some example problems and solutions:\n\n"
                     f"{ctx}<|im_end|>\n")
        parts.append(f"<|im_start|>assistant\nUnderstood.<|im_end|>\n")
    user_text = question.strip()
    if starter_code and starter_code.strip():
        user_text += f"\n\nStarter code:\n{starter_code.strip()}"
    parts.append(f"<|im_start|>user\n{user_text}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def _strip_fences(text: str) -> str:
    """Remove markdown code fences if the model added them anyway."""
    text = re.sub(r"^```(?:python)?\s*\n?", "", text.strip())
    text = re.sub(r"\n?```\s*$", "", text).strip()
    return text


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return "unknown"


# ── Main eval loop ───────────────────────────────────────────────────────────

def run_eval(
    model_path: str,
    test_path: Path,
    out_dir: Path,
    backend: str = "mlx",
    draft_model_path: str | None = None,
    num_draft_tokens: int = 5,
    mode: str = "no-context",
    train_path: Path | None = None,
    n_shots: int = 3,
    limit: int | None = None,
    max_tokens: int = MAX_OUTPUT_TOKENS,
    temp: float = 0.0,
    skip_verify: bool = False,
    exec_timeout: float = 10.0,
    max_cases: int = 5,
    seed: int = 42,
) -> dict:

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load test records
    records: list[dict] = []
    with test_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if limit is not None:
        records = records[:limit]

    # Load context examples for with-context mode
    context_examples_all: list | None = None
    if mode == "with-context":
        if train_path is None or not train_path.exists():
            sys.exit("--train is required for --mode with-context")
        from apps_context import load_train_examples
        context_examples_all = load_train_examples(train_path)

    # Build backend
    if backend == "mock":
        generate_fn, count_tokens_fn = _make_mock_backend()
    elif backend == "hf":
        generate_fn, count_tokens_fn = _make_hf_backend(model_path, draft_model_path)
    else:
        generate_fn, count_tokens_fn = _make_mlx_backend(model_path, draft_model_path)

    # Write run_config
    run_cfg = {
        "model": model_path,
        "backend": backend,
        "draft_model": draft_model_path,
        "num_draft_tokens": num_draft_tokens if draft_model_path else None,
        "mode": mode,
        "n_shots": n_shots if mode == "with-context" else 0,
        "test_path": str(test_path),
        "train_path": str(train_path) if train_path else None,
        "limit": limit,
        "max_tokens": max_tokens,
        "temp": temp,
        "skip_verify": skip_verify,
        "exec_timeout": exec_timeout,
        "max_cases": max_cases,
        "seed": seed,
        "git_commit": _git_commit(),
        "platform": platform.platform(),
        "python": sys.version,
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2))

    pred_path = out_dir / "predictions.jsonl"
    fail_path = out_dir / "failures.jsonl"
    (out_dir / "summary.json").unlink(missing_ok=True)

    n_pass = n_fail = n_skip = n_scan_blocked = 0
    total_tokens = total_elapsed = 0.0

    with pred_path.open("w") as pf, fail_path.open("w") as ff:
        for i, rec in enumerate(records):
            question   = rec.get("question", "")
            starter    = rec.get("starter_code", "")
            io         = rec.get("input_output", {})
            difficulty = rec.get("difficulty", "")
            prob_id    = rec.get("problem_id")

            # Context retrieval
            ctx_examples = None
            if mode == "with-context" and context_examples_all is not None:
                from apps_context import select_context_examples
                ctx_examples = select_context_examples(
                    question, context_examples_all, n_shots,
                    same_difficulty=difficulty or None,
                )

            prompt = _build_prompt(question, starter, ctx_examples)
            t0 = time.monotonic()
            raw_output, elapsed = generate_fn(prompt, max_tokens, num_draft_tokens, temp)
            code = _strip_fences(raw_output)
            n_toks = count_tokens_fn(raw_output)
            tps = n_toks / elapsed if elapsed > 0 else 0.0
            total_tokens += n_toks
            total_elapsed += elapsed

            # Verify
            passed = scan_blocked = False
            scan_reason = stderr = ""
            n_cases_passed = n_cases_total = 0
            timed_out = False

            if skip_verify or not io:
                n_skip += 1
            else:
                from apps_verify import execute_solution
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
                "problem_id": prob_id,
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
                "tokens": n_toks,
                "tps": round(tps, 1),
            }
            pf.write(json.dumps(row, ensure_ascii=False) + "\n")
            if not passed and not scan_blocked:
                rec_out = dict(rec)
                rec_out["generated_code"] = code
                rec_out["stderr"] = stderr[:400]
                ff.write(json.dumps(rec_out, ensure_ascii=False) + "\n")

            eligible = n_pass + n_fail
            rate_str = f"{n_pass/eligible*100:.1f}%" if eligible > 0 else "?"
            print(
                f"[{i+1}/{len(records)}] {difficulty} pass={rate_str} "
                f"tps={tps:.0f} scan_blocked={scan_blocked}",
                flush=True,
            )

    eligible = n_pass + n_fail
    summary = {
        "model": model_path,
        "backend": backend,
        "draft_model": draft_model_path,
        "num_draft_tokens": num_draft_tokens if draft_model_path else None,
        "mode": mode,
        "test_path": str(test_path),
        "n_total": len(records),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_skip": n_skip,
        "n_scan_blocked": n_scan_blocked,
        "pass_rate": n_pass / eligible if eligible > 0 else None,
        "mean_elapsed_s": round(total_elapsed / len(records), 3) if records else 0,
        "mean_tps": round(total_tokens / total_elapsed, 1) if total_elapsed > 0 else 0,
        "total_tokens": int(total_tokens),
        "total_elapsed_s": round(total_elapsed, 1),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", required=True,
                   help="Fused MLX dir (mlx), PEFT adapter dir or HF model name (hf)")
    p.add_argument("--backend", choices=["mlx", "hf", "mock"], default="mlx")
    p.add_argument("--draft-model", default=None)
    p.add_argument("--num-draft-tokens", type=int, default=5)
    p.add_argument("--mode", choices=["no-context", "with-context"], default="no-context")
    p.add_argument("--train", type=Path, default=None,
                   help="Training JSONL for context retrieval (with-context mode)")
    p.add_argument("--n-shots", type=int, default=3)
    p.add_argument("--test", type=Path, default=Path("data/apps/eval_all.jsonl"))
    p.add_argument("--output", type=Path, default=Path("results/apps/default"))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=MAX_OUTPUT_TOKENS)
    p.add_argument("--temp", type=float, default=0.0)
    p.add_argument("--skip-verify", action="store_true",
                   help="Skip code execution (safety scan still runs)")
    p.add_argument("--exec-timeout", type=float, default=10.0,
                   help="Per-test-case execution timeout in seconds")
    p.add_argument("--max-cases", type=int, default=5,
                   help="Max test cases to run per problem")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    summary = run_eval(
        model_path=args.model,
        test_path=args.test,
        out_dir=args.output,
        backend=args.backend,
        draft_model_path=args.draft_model,
        num_draft_tokens=args.num_draft_tokens,
        mode=args.mode,
        train_path=args.train,
        n_shots=args.n_shots,
        limit=args.limit,
        max_tokens=args.max_tokens,
        temp=args.temp,
        skip_verify=args.skip_verify,
        exec_timeout=args.exec_timeout,
        max_cases=args.max_cases,
        seed=args.seed,
    )
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
