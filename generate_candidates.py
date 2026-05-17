#!/usr/bin/env python3
"""
Rejection sampling: generate N candidates per prompt, score, keep the best.

Queries the fine-tuned model via vLLM with temperature > 0 to produce diverse
completions, then scores each candidate against the reference using composite
scoring (structural + similarity).  Outputs best-of-N pairs as ShareGPT JSONL
suitable for another SFT round or DPO (with best/worst pairs).

Usage:
    python generate_candidates.py                          # defaults from config
    python generate_candidates.py --n 8 --temperature 0.7
    python generate_candidates.py --output-dpo            # emit chosen/rejected pairs

Env overrides:
    VLLM_URL, MODEL, HOLDOUT, CANDIDATES_OUT, CANDIDATES_N, CANDIDATES_TEMP
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import _config
from _eval_utils import similarity, structural_score, composite_score, band


def generate_candidates(client, model: str, messages: list[dict], n: int,
                        temperature: float, max_tokens: int) -> list[str]:
    """Generate n candidate responses for a prompt."""
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        n=n,
        seed=None,
    )
    return [c.message.content or "" for c in resp.choices]


def score_candidate(generated: str, expected: str) -> float:
    sim = similarity(expected, generated)
    struct = structural_score(expected, generated)
    return composite_score(sim, struct)


def process_record(client, model: str, record: dict, n: int,
                   temperature: float, max_tokens: int) -> dict:
    """Generate candidates for one record, score all, return best + metadata."""
    messages = [{"role": m["role"], "content": m["content"]}
                for m in record["messages"] if m["role"] != "assistant"]
    expected = next(
        (m["content"] for m in record["messages"] if m["role"] == "assistant"), ""
    )

    candidates = generate_candidates(client, model, messages, n, temperature, max_tokens)
    scored = []
    for c in candidates:
        s = score_candidate(c, expected)
        scored.append((s, c))
    scored.sort(key=lambda x: x[0], reverse=True)

    best_score, best_text = scored[0]
    worst_score, worst_text = scored[-1]

    return {
        "id": record.get("id", ""),
        "label": record.get("label", ""),
        "prompt": messages,
        "expected": expected,
        "best": best_text,
        "best_score": round(best_score, 4),
        "worst": worst_text,
        "worst_score": round(worst_score, 4),
        "n_candidates": len(candidates),
        "mean_score": round(sum(s for s, _ in scored) / len(scored), 4),
        "band": band(best_score),
    }


def to_sharegpt(prompt_messages: list[dict], response: str) -> dict:
    """Convert to ShareGPT format for SFT training."""
    conversations = []
    for m in prompt_messages:
        role_map = {"system": "system", "user": "human"}
        conversations.append({"from": role_map.get(m["role"], m["role"]), "value": m["content"]})
    conversations.append({"from": "gpt", "value": response})
    return {"conversations": conversations}


def to_dpo_pair(result: dict) -> dict:
    """Convert to DPO preference pair format."""
    conversations = []
    for m in result["prompt"]:
        role_map = {"system": "system", "user": "human"}
        conversations.append({"from": role_map.get(m["role"], m["role"]), "value": m["content"]})
    return {
        "conversations": conversations,
        "chosen": result["best"],
        "rejected": result["worst"],
        "chosen_score": result["best_score"],
        "rejected_score": result["worst_score"],
    }


def main() -> None:
    from openai import OpenAI

    cfg = _config.load()

    parser = argparse.ArgumentParser(description="Rejection sampling via best-of-N")
    parser.add_argument("--n", type=int,
                        default=int(os.environ.get("CANDIDATES_N", "4")))
    parser.add_argument("--temperature", type=float,
                        default=float(os.environ.get("CANDIDATES_TEMP", "0.6")))
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--min-score", type=float, default=0.6,
                        help="Minimum composite score to include in output")
    parser.add_argument("--output-dpo", action="store_true",
                        help="Emit DPO chosen/rejected pairs instead of SFT")
    parser.add_argument("--input", type=str, default=None,
                        help="Input JSONL (default: holdout from config)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: data_dir/rejection_sampled.jsonl)")
    args = parser.parse_args()

    BASE_URL = os.environ.get("VLLM_URL", cfg.vllm_url)
    MODEL = os.environ.get("MODEL", cfg.adapter_name)
    INPUT = Path(args.input or os.environ.get("HOLDOUT", str(cfg.holdout))).expanduser()
    OUTPUT = Path(args.output or os.environ.get(
        "CANDIDATES_OUT", str(cfg.data_dir / "rejection_sampled.jsonl")))

    client = OpenAI(base_url=f"{BASE_URL}/v1", api_key="none")

    with open(INPUT) as fh:
        records = [json.loads(line) for line in fh]

    print(f"Model:       {MODEL}")
    print(f"Server:      {BASE_URL}")
    print(f"Input:       {INPUT} ({len(records)} records)")
    print(f"Candidates:  {args.n} per prompt")
    print(f"Temperature: {args.temperature}")
    print(f"Min score:   {args.min_score}")
    print(f"Output mode: {'DPO pairs' if args.output_dpo else 'SFT (best-of-N)'}")
    print("=" * 60)

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_record, client, MODEL, r, args.n,
                        args.temperature, args.max_tokens): i
            for i, r in enumerate(records)
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                result = fut.result()
                results.append(result)
                print(f"  [{result['band']:9s} {result['best_score']:.2f} "
                      f"avg={result['mean_score']:.2f}]  "
                      f"{result['id']}  {result['label']}")
            except Exception as e:
                print(f"  [ERROR]  record {idx}: {e}")

    # Filter and write output
    kept = [r for r in results if r["best_score"] >= args.min_score]
    kept.sort(key=lambda r: r["best_score"], reverse=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as fh:
        for r in kept:
            if args.output_dpo:
                fh.write(json.dumps(to_dpo_pair(r)) + "\n")
            else:
                fh.write(json.dumps(to_sharegpt(r["prompt"], r["best"])) + "\n")

    # Summary
    print("=" * 60)
    print(f"Total processed: {len(results)}")
    print(f"Kept (score >= {args.min_score}): {len(kept)}")
    if kept:
        scores = [r["best_score"] for r in kept]
        print(f"Score range: {min(scores):.3f} - {max(scores):.3f}")
        print(f"Mean best score: {sum(scores)/len(scores):.3f}")
    print(f"Output: {OUTPUT}")

    # Write metadata alongside
    meta_path = OUTPUT.with_suffix(".meta.json")
    meta = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL,
        "n_candidates": args.n,
        "temperature": args.temperature,
        "min_score": args.min_score,
        "input_records": len(records),
        "output_records": len(kept),
        "output_mode": "dpo" if args.output_dpo else "sft",
        "mean_best_score": round(sum(r["best_score"] for r in kept) / max(1, len(kept)), 4),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
