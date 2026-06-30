#!/usr/bin/env python3
"""
Lean tactic evaluation runner.

Reads test.jsonl, generates one tactic per record using an LLM, safety-checks
each tactic, optionally verifies with Lean, and writes per-record prediction
JSONL + compiler_errors.jsonl + run_config.json + summary JSON.  Output field
names match the spec in FULL_EVAL.md.

Backends
--------
  mlx   (default)  Apple MLX — requires Mac and mlx-lm>=0.21.0
  hf               HuggingFace/Unsloth — for GB10 validation; accepts PEFT
                   adapter paths or base HF model names
  mock             Returns predetermined tactics (no model needed); use to
                   validate the full pipeline — data loading, Lean verification,
                   output format, make_report.py — without loading any weights

Usage:
  # Production (Mac)
  python3 scripts/lean_eval.py --model fused-7b-lean \\
      --test data/lean_stat/test.jsonl \\
      --output reports/lean_eval/target-7b

  # Speculative decoding (Mac)
  python3 scripts/lean_eval.py --model fused-7b-lean \\
      --draft-model fused-0.5b-lean --num-draft-tokens 5 \\
      --test data/lean_stat/test.jsonl \\
      --output reports/lean_eval/speculative

  # GB10 pipeline validation (HF/PEFT)
  python3 lean_eval.py --backend hf \\
      --model adapters/7b-lean/final \\
      --test data/lean_stat/test.jsonl --limit 20 \\
      --output reports/lean_eval/hf-target-7b

  # Pipeline-only smoke test (no model, no GPU)
  python3 lean_eval.py --backend mock \\
      --model unused --limit 20 \\
      --test data/lean_stat/test.jsonl \\
      --output reports/lean_eval/mock
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
    "You are a Lean 4 proof assistant. "
    "Given a tactic proof state, provide exactly one tactic that makes progress."
)

CHAT_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{user}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

# Cycling tactics for the mock backend — cover common Lean proof patterns
_MOCK_TACTICS = [
    "simp", "rfl", "omega", "trivial", "exact h",
    "ring", "linarith", "norm_num", "exact?", "apply?",
]


# ---------------------------------------------------------------------------
# Backend factories — each returns (generate_fn, count_tokens_fn)
# ---------------------------------------------------------------------------

def _make_mock_backend():
    counter = [0]

    def generate_fn(prompt: str, max_tokens: int, num_draft_tokens: int,
                    temp: float) -> tuple[str, float]:
        tactic = _MOCK_TACTICS[counter[0] % len(_MOCK_TACTICS)]
        counter[0] += 1
        return tactic, 0.001

    def count_tokens_fn(text: str) -> int:
        return len(text.split())

    return generate_fn, count_tokens_fn


def _make_hf_backend(model_path: str, draft_model_path: str | None):
    import json as _json
    import os
    import tempfile
    import shutil

    # Ensure HF_HOME points to the local cache
    if not os.environ.get("HF_HOME"):
        candidate = Path.home() / "llm" / "models" / "hf"
        if candidate.exists():
            os.environ["HF_HOME"] = str(candidate)

    def _resolve_adapter(path: str) -> str:
        """If path is a PEFT adapter with a HF repo name as base, patch it."""
        p = Path(path)
        cfg_path = p / "adapter_config.json"
        if not cfg_path.exists():
            return path  # base model path, pass through

        cfg = _json.loads(cfg_path.read_text())
        base = cfg.get("base_model_name_or_path", "")
        if Path(base).exists():
            return path  # already a local path

        # base is a HF repo name — resolve via local cache
        try:
            from huggingface_hub import snapshot_download
            local_base = snapshot_download(base, local_files_only=True)
        except Exception as exc:
            sys.exit(
                f"Cannot resolve base model '{base}' from local cache.\n"
                f"Set HF_HOME to your model cache or download the model first.\n"
                f"Error: {exc}"
            )
        tmp = tempfile.mkdtemp(prefix="lean_eval_adapter_")
        shutil.copytree(str(p), tmp + "/adapter", dirs_exist_ok=True)
        cfg["base_model_name_or_path"] = local_base
        (Path(tmp) / "adapter" / "adapter_config.json").write_text(
            _json.dumps(cfg, indent=2))
        return tmp + "/adapter"

    try:
        from unsloth import FastLanguageModel
    except ImportError:
        sys.exit("unsloth not found. Install it or use --backend mlx on Mac.")

    import torch

    resolved = _resolve_adapter(model_path)
    model, tokenizer = FastLanguageModel.from_pretrained(
        resolved, max_seq_length=32768, load_in_4bit=False, dtype=None,
    )
    FastLanguageModel.for_inference(model)

    # Draft model is not supported in HF backend (speculative needs MLX)
    if draft_model_path:
        print("WARNING: --draft-model ignored for --backend hf "
              "(speculative decoding requires MLX)", file=sys.stderr)

    def generate_fn(prompt: str, max_tokens: int, num_draft_tokens: int,
                    temp: float) -> tuple[str, float]:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        t0 = time.monotonic()
        do_sample = temp > 0
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temp if do_sample else None,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.monotonic() - t0
        text = tokenizer.decode(
            out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        return text.split("<|im_end|>")[0].strip(), elapsed

    def count_tokens_fn(text: str) -> int:
        ids = tokenizer.encode(text)
        return len(ids) if isinstance(ids, list) else ids.shape[-1]

    return generate_fn, count_tokens_fn


def _make_mlx_backend(model_path: str, draft_model_path: str | None):
    try:
        import importlib.metadata
        from packaging.version import Version
        ver = importlib.metadata.version("mlx-lm")
        if Version(ver) < Version(MLX_LM_MIN_VERSION):
            print(f"WARNING: mlx-lm {ver} < required {MLX_LM_MIN_VERSION}. "
                  f"Upgrade: pip install 'mlx-lm>={MLX_LM_MIN_VERSION}'",
                  file=sys.stderr)
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
    _verbose_supported = "verbose" in inspect.signature(mlx_generate).parameters

    def generate_fn(prompt: str, max_tokens: int, num_draft_tokens: int,
                    temp: float) -> tuple[str, float]:
        t0 = time.monotonic()
        kwargs: dict = dict(max_tokens=max_tokens, temp=temp)
        if _verbose_supported:
            kwargs["verbose"] = False
        if draft is not None:
            kwargs["draft_model"] = draft
            kwargs["num_draft_tokens"] = num_draft_tokens
        response = mlx_generate(model, tokenizer, prompt=prompt, **kwargs)
        elapsed = time.monotonic() - t0
        return response.split("<|im_end|>")[0].strip(), elapsed

    def count_tokens_fn(text: str) -> int:
        ids = tokenizer.encode(text)
        return len(ids) if isinstance(ids, list) else ids.shape[-1]

    return generate_fn, count_tokens_fn


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _user_prompt(state_before: str) -> str:
    return f"Given the Lean 4 state:\n{state_before}\nProvide the next tactical step."


def _build_prompt(state_before: str, context_examples: list | None = None) -> str:
    context_examples = context_examples or []
    parts = [f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"]
    for example in context_examples:
        parts.append(f"<|im_start|>user\n{_user_prompt(example.state_before)}<|im_end|>\n")
        parts.append(f"<|im_start|>assistant\n{example.tactic}<|im_end|>\n")
    parts.append(f"<|im_start|>user\n{_user_prompt(state_before)}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_eval(
    model_path: str,
    test_path: Path,
    out_dir: Path,
    backend: str = "mlx",
    draft_model_path: str | None = None,
    num_draft_tokens: int = 5,
    max_tokens: int = 256,
    temp: float = 0.0,
    limit: int | None = None,
    lean_timeout: int = 60,
    skip_lean: bool = False,
    context_train_path: Path | None = None,
    n_shots: int = 0,
    max_context_state_chars: int = 1200,
    lean_project: Path | None = None,
    max_state_chars: int | None = None,
) -> dict:
    sys.path.insert(0, str(Path(__file__).parent))
    from lean_verify import safety_check, verify_tactic
    from lean_context import extract_state_tactic, load_examples, select_context_examples

    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    with open(test_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if limit is not None:
        records = records[:limit]

    context_pool = []
    if context_train_path is not None and n_shots > 0:
        context_pool = load_examples(context_train_path)

    run_cfg = {
        "model": model_path,
        "backend": backend,
        "draft_model": draft_model_path,
        "num_draft_tokens": num_draft_tokens if draft_model_path else None,
        "test_path": str(test_path),
        "max_tokens": max_tokens,
        "temp": temp,
        "limit": limit,
        "skip_lean": skip_lean,
        "lean_project": str(lean_project) if lean_project else None,
        "context_train_path": str(context_train_path) if context_train_path else None,
        "n_shots": n_shots,
        "context_retrieval": "lexical_jaccard" if context_pool else None,
        "max_context_state_chars": max_context_state_chars if context_pool else None,
        "max_state_chars": max_state_chars,
        "git_commit": _git_commit(),
        "platform": platform.platform(),
        "python": sys.version,
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2))

    if backend == "mock":
        generate_fn, count_tokens_fn = _make_mock_backend()
    elif backend == "hf":
        generate_fn, count_tokens_fn = _make_hf_backend(model_path, draft_model_path)
    else:
        generate_fn, count_tokens_fn = _make_mlx_backend(model_path, draft_model_path)

    pred_path = out_dir / "predictions.jsonl"
    err_path = out_dir / "compiler_errors.jsonl"

    n_safety_fail = 0
    n_lean_pass = 0
    n_lean_fail = 0
    n_lean_skip = 0
    total_tokens = 0
    total_elapsed = 0.0

    n_skipped_size = 0
    with open(pred_path, "w") as pred_f, open(err_path, "w") as err_f:
        for i, rec in enumerate(records):
            example = extract_state_tactic(rec, index=i)
            state_before = example.state_before
            expected_tactic = example.tactic

            if max_state_chars is not None and len(state_before) > max_state_chars:
                n_skipped_size += 1
                continue

            context_examples = select_context_examples(
                state_before,
                context_pool,
                n_shots=n_shots,
                max_state_chars=max_context_state_chars,
            )

            prompt = _build_prompt(state_before, context_examples)
            gen_text, elapsed = generate_fn(prompt, max_tokens, num_draft_tokens, temp)
            n_tokens = count_tokens_fn(gen_text)
            tps = n_tokens / elapsed if elapsed > 0 else 0.0
            total_tokens += n_tokens
            total_elapsed += elapsed

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
                "prompt": prompt,
                "context_examples": [ex.to_dict() for ex in context_examples],
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
    n_evaluated = n_total - n_skipped_size
    lean_eligible = n_lean_pass + n_lean_fail
    summary = {
        "model": model_path,
        "backend": backend,
        "draft_model": draft_model_path,
        "num_draft_tokens": num_draft_tokens if draft_model_path else None,
        "test_path": str(test_path),
        "n_total": n_total,
        "n_skipped_size": n_skipped_size,
        "n_evaluated": n_evaluated,
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
                   help="Model path: fused MLX dir (mlx), PEFT adapter dir or HF "
                        "repo name (hf), or any string (mock)")
    p.add_argument("--backend", choices=["mlx", "hf", "mock"], default="mlx",
                   help="Inference backend (default: mlx)")
    p.add_argument("--draft-model", default=None,
                   help="Draft model path for speculative decoding (mlx backend only)")
    p.add_argument("--num-draft-tokens", type=int, default=5)
    p.add_argument("--test", type=Path, default=Path("data/lean_stat/test.jsonl"))
    p.add_argument("--output", type=Path, default=Path("reports/lean_eval/run"))
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temp", type=float, default=0.0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--lean-timeout", type=int, default=60)
    p.add_argument("--skip-lean", action="store_true")
    p.add_argument("--lean-project", type=Path, default=None,
                   help="Path to a Lake project root (e.g. ~/git_home/lean-stat-learning-theory). "
                        "When set, compilation uses `lake env lean` so SLT/Mathlib identifiers resolve.")
    p.add_argument("--context-train", type=Path, default=None,
                   help="Training JSONL used to retrieve few-shot context examples")
    p.add_argument("--n-shots", type=int, default=0,
                   help="Number of retrieved examples to add before each query")
    p.add_argument("--max-context-state-chars", type=int, default=1200,
                   help="Skip context candidates with states longer than this")
    p.add_argument("--max-state-chars", type=int, default=None,
                   help="Skip evaluation records where the state exceeds this many chars "
                        "(avoids OOM on deeply nested Lean terms)")
    args = p.parse_args()

    summary = run_eval(
        model_path=args.model,
        test_path=args.test,
        out_dir=args.output,
        backend=args.backend,
        draft_model_path=args.draft_model,
        num_draft_tokens=args.num_draft_tokens,
        max_tokens=args.max_tokens,
        temp=args.temp,
        limit=args.limit,
        lean_timeout=args.lean_timeout,
        skip_lean=args.skip_lean,
        lean_project=args.lean_project,
        context_train_path=args.context_train,
        n_shots=args.n_shots,
        max_context_state_chars=args.max_context_state_chars,
        max_state_chars=args.max_state_chars,
    )
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
