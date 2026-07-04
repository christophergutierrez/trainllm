#!/usr/bin/env python3
"""Evaluate pass@1 on MBPP/MBPP+ JSONL tasks."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from code_verify import DEFAULT_TIMEOUT, clean_completion

SYSTEM = (
    "You are an expert Python programmer. Write a complete, correct Python "
    "solution. Output only executable Python code, with no markdown fences or "
    "explanation."
)


@dataclass
class VerifyResult:
    passed: bool
    stderr: str = ""
    timed_out: bool = False


def _make_hf_backend(model_path: str):
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise SystemExit("transformers/peft/torch not found.")

    p = Path(model_path).expanduser()
    adapter_cfg = p / "adapter_config.json"
    device_map = {"": 0} if torch.cuda.is_available() else "cpu"

    if adapter_cfg.exists():
        base_name = json.loads(adapter_cfg.read_text())["base_model_name_or_path"]
        tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            base_name, dtype=torch.float16, device_map=device_map, trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base, str(p)).merge_and_unload()
    else:
        tokenizer = AutoTokenizer.from_pretrained(str(p), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(p), dtype=torch.float16, device_map=device_map, trust_remote_code=True
        )

    model.eval()

    def generate(prompt: str, max_tokens: int, temperature: float) -> tuple[str, float]:
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        n_in = inputs["input_ids"].shape[1]
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - t0
        n_out = out.shape[1] - n_in
        return tokenizer.decode(out[0][n_in:], skip_special_tokens=True), n_out / elapsed if elapsed else 0

    return generate


def verify_python(completion: str, test_code: str, timeout: float = DEFAULT_TIMEOUT) -> VerifyResult:
    script = clean_completion(completion).strip() + "\n\n" + test_code
    script_path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(script)
            script_path = f.name
        proc = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return VerifyResult(False, "timeout", True)
    finally:
        if script_path:
            try:
                Path(script_path).unlink()
            except OSError:
                pass
    if proc.returncode == 0:
        return VerifyResult(True)
    return VerifyResult(False, proc.stderr[:500])


def run_eval(
    model: str,
    test: Path,
    output: Path,
    max_tokens: int,
    temperature: float,
    limit: int | None,
) -> dict:
    problems = [json.loads(line) for line in test.read_text().splitlines() if line.strip()]
    if limit:
        problems = problems[:limit]

    generate = _make_hf_backend(model)
    output.mkdir(parents=True, exist_ok=True)
    pred_path = output / "predictions.jsonl"
    fail_path = output / "failures.jsonl"

    passed = 0
    tps_values: list[float] = []
    with pred_path.open("w") as pf, fail_path.open("w") as ff:
        for idx, problem in enumerate(problems, 1):
            completion, tps = generate(problem["prompt"], max_tokens, temperature)
            completion = clean_completion(completion)
            result = verify_python(completion, problem["test"])
            passed += int(result.passed)
            tps_values.append(tps)
            rec = {
                "task_id": problem["task_id"],
                "passed": result.passed,
                "completion": completion[:2000],
                "stderr": result.stderr[:200],
                "timed_out": result.timed_out,
                "tps": round(tps, 1),
            }
            pf.write(json.dumps(rec) + "\n")
            if not result.passed:
                ff.write(json.dumps(rec) + "\n")
            print(
                f"[{idx}/{len(problems)}] {problem['task_id']} "
                f"pass={passed / idx:.1%} tps={tps:.0f}",
                flush=True,
            )

    total = len(problems)
    summary = {
        "model": model,
        "test_path": str(test),
        "n_total": total,
        "n_pass": passed,
        "n_fail": total - passed,
        "pass_rate": passed / total if total else 0,
        "mean_tps": round(sum(tps_values) / len(tps_values), 1) if tps_values else 0,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "run_config.json").write_text(
        json.dumps({"model": model, "test": str(test), "max_tokens": max_tokens, "limit": limit}, indent=2) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--test", type=Path, default=Path("data/mbpp/mbpp_plus_test.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    summary = run_eval(args.model, args.test, args.output, args.max_tokens, args.temp, args.limit)
    print(f"\npass@1 = {summary['pass_rate']:.1%} ({summary['n_pass']}/{summary['n_total']})")
    print(f"mean tps = {summary['mean_tps']}")


if __name__ == "__main__":
    main()
