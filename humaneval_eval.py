#!/usr/bin/env python3
"""
Evaluate pass@1 on HumanEval using a local model (HF or MLX backend).

Sources:
  Test set: data/magicoder/humaneval_plus_test.jsonl  (164 problems)
  Model:    a LoRA adapter path or a base model name

Usage:
    # Fine-tuned 7B
    python3 humaneval_eval.py --backend hf --model lora/magicoder-7b/final \
        --output results/magicoder/target-7b

    # Base model baseline
    python3 humaneval_eval.py --backend hf --model Qwen/Qwen2.5-Coder-7B-Instruct \
        --output results/magicoder/base-7b

    # Speculative (MLX, Mac)
    python3 humaneval_eval.py --backend mlx --model adapters/7b-magicoder-mlx \
        --draft-model adapters/0.5b-magicoder-mlx \
        --output results/magicoder/speculative
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


# ── Prompt builder ────────────────────────────────────────────────────────────

SYSTEM = (
    "You are an expert Python programmer. "
    "Complete the Python function below. Output the complete function definition "
    "(including the def line, docstring, and body) — no markdown fences, no explanation."
)


def build_messages(prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": prompt},
    ]


# ── HF backend ────────────────────────────────────────────────────────────────

def _make_hf_backend(model_path: str):
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
    except ImportError:
        sys.exit("transformers/peft/torch not found.")

    p = Path(model_path).expanduser()
    cfg_path = p / "adapter_config.json"
    device_map = {"": 0} if __import__("torch").cuda.is_available() else "cpu"

    if cfg_path.exists():
        base_name = json.loads(cfg_path.read_text())["base_model_name_or_path"]
        tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            base_name, dtype=torch.float16, device_map=device_map, trust_remote_code=True)
        model = PeftModel.from_pretrained(base, str(p)).merge_and_unload()
    else:
        tokenizer = AutoTokenizer.from_pretrained(str(p), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(p), dtype=torch.float16, device_map=device_map, trust_remote_code=True)

    model.eval()

    def generate(messages: list[dict], max_tokens: int, temperature: float) -> tuple[str, float]:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        n_in = inputs["input_ids"].shape[1]
        t0 = time.perf_counter()
        with __import__("torch").no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - t0
        n_out = out.shape[1] - n_in
        tps = n_out / elapsed if elapsed > 0 else 0
        completion = tokenizer.decode(out[0][n_in:], skip_special_tokens=True)
        return completion, tps

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text))

    return generate, count_tokens


# ── MLX backend ───────────────────────────────────────────────────────────────

def _make_mlx_backend(model_path: str, draft_model_path: str | None, num_draft_tokens: int):
    try:
        from mlx_lm import load, generate as mlx_generate
    except ImportError:
        sys.exit("mlx_lm not found: pip install mlx-lm")

    model, tokenizer = load(model_path)
    draft = None
    if draft_model_path:
        draft, _ = load(draft_model_path)

    def generate(messages: list[dict], max_tokens: int, temperature: float) -> tuple[str, float]:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        t0 = time.perf_counter()
        kwargs = {"max_tokens": max_tokens, "temp": temperature}
        if draft:
            kwargs["draft_model"] = draft
            kwargs["num_draft_tokens"] = num_draft_tokens
        out = mlx_generate(model, tokenizer, prompt=prompt, **kwargs)
        elapsed = time.perf_counter() - t0
        n_tokens = len(tokenizer.encode(out))
        tps = n_tokens / elapsed if elapsed > 0 else 0
        return out, tps

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text))

    return generate, count_tokens


# ── Completion cleanup ────────────────────────────────────────────────────────

def _strip_fences(text: str) -> str:
    """Extract executable code from common fenced model output."""
    from code_verify import clean_completion

    return clean_completion(text)


# ── Eval loop ─────────────────────────────────────────────────────────────────

def run_eval(
    model_path: str,
    test_path: Path,
    output_dir: Path,
    backend: str,
    draft_model_path: str | None,
    num_draft_tokens: int,
    max_tokens: int,
    temperature: float,
    limit: int | None,
) -> dict:
    from code_verify import verify_humaneval

    problems = [json.loads(l) for l in test_path.read_text().splitlines() if l.strip()]
    if limit:
        problems = problems[:limit]

    if backend == "hf":
        generate, count_tokens = _make_hf_backend(model_path)
    else:
        generate, count_tokens = _make_mlx_backend(model_path, draft_model_path, num_draft_tokens)

    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / "predictions.jsonl"
    fail_path = output_dir / "failures.jsonl"

    n_pass = n_fail = 0
    total_tokens = total_elapsed = 0.0
    tps_list = []

    with pred_path.open("w") as pf, fail_path.open("w") as ff:
        for i, prob in enumerate(problems, 1):
            messages  = build_messages(prob["prompt"])
            completion, tps = generate(messages, max_tokens, temperature)
            completion = _strip_fences(completion)

            result = verify_humaneval(
                prob["prompt"], completion, prob["test"],
                entry_point=prob.get("entry_point", ""),
            )

            rec = {
                "task_id":    prob["task_id"],
                "passed":     result.passed,
                "completion": completion[:2000],
                "tps":        round(tps, 1),
                "timed_out":  result.timed_out,
                "stderr":     result.stderr[:200] if result.stderr else "",
            }
            pf.write(json.dumps(rec) + "\n")
            if not result.passed:
                ff.write(json.dumps(rec) + "\n")

            if result.passed:
                n_pass += 1
            else:
                n_fail += 1

            tps_list.append(tps)
            total_elapsed += (1 / tps) * max_tokens if tps > 0 else 0

            running_rate = n_pass / i
            print(f"[{i}/{len(problems)}] {prob['task_id']} "
                  f"pass={running_rate:.1%} tps={tps:.0f}", flush=True)

    mean_tps = sum(tps_list) / len(tps_list) if tps_list else 0
    summary = {
        "model":           model_path,
        "backend":         backend,
        "draft_model":     draft_model_path,
        "num_draft_tokens": num_draft_tokens if draft_model_path else None,
        "test_path":       str(test_path),
        "n_total":         n_pass + n_fail,
        "n_pass":          n_pass,
        "n_fail":          n_fail,
        "pass_rate":       n_pass / (n_pass + n_fail) if (n_pass + n_fail) else 0,
        "mean_tps":        round(mean_tps, 1),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "run_config.json").write_text(json.dumps(
        {"model": model_path, "backend": backend, "max_tokens": max_tokens,
         "temperature": temperature, "limit": limit}, indent=2))
    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model",      required=True)
    p.add_argument("--test",       type=Path, default=Path("data/magicoder/humaneval_plus_test.jsonl"))
    p.add_argument("--output",     type=Path, required=True)
    p.add_argument("--backend",    choices=["hf", "mlx"], default="hf")
    p.add_argument("--draft-model",     default=None)
    p.add_argument("--num-draft-tokens", type=int, default=5)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temp",       type=float, default=0.0)
    p.add_argument("--limit",      type=int, default=None,
                   help="Evaluate only first N problems (smoke test)")
    args = p.parse_args()

    summary = run_eval(
        model_path=args.model,
        test_path=args.test,
        output_dir=args.output,
        backend=args.backend,
        draft_model_path=args.draft_model,
        num_draft_tokens=args.num_draft_tokens,
        max_tokens=args.max_tokens,
        temperature=args.temp,
        limit=args.limit,
    )
    print(f"\npass@1 = {summary['pass_rate']:.1%}  ({summary['n_pass']}/{summary['n_total']})")
    print(f"mean tps = {summary['mean_tps']}")


if __name__ == "__main__":
    main()
