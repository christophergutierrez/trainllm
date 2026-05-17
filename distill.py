#!/usr/bin/env python3
"""
Distillation: generate training data from a stronger teacher model.

Takes prompts from training/holdout data and queries a teacher model to produce
high-quality responses.  Supports two teacher backends:
  1. Anthropic API (Claude) — requires ANTHROPIC_API_KEY
  2. OpenAI-compatible endpoint (vLLM with a larger model, or OpenAI API)

The output is ShareGPT JSONL ready for SFT training.

Usage:
    # Using Claude as teacher
    ANTHROPIC_API_KEY=sk-... python distill.py --teacher anthropic

    # Using a local vLLM-served teacher
    python distill.py --teacher vllm --teacher-model Qwen/Qwen2.5-72B-Instruct

    # Using OpenAI API
    OPENAI_API_KEY=sk-... python distill.py --teacher openai --teacher-model gpt-4o

Env overrides:
    ANTHROPIC_API_KEY, OPENAI_API_KEY, TEACHER_URL, DISTILL_OUT
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import _config


def _query_anthropic(client, model: str, system: str, user_content: str,
                     max_tokens: int) -> str:
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user_content}],
    )
    text_block = next((b for b in resp.content if b.type == "text"), None)
    if text_block is None:
        raise RuntimeError("No text block in Anthropic response")
    return text_block.text


def _query_openai(client, model: str, system: str, user_content: str,
                  max_tokens: int) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_content})
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return resp.choices[0].message.content or ""


def _extract_prompt_parts(record: dict) -> tuple[str, str]:
    """Extract system prompt and user content from a record.

    Handles both ShareGPT format (conversations) and OpenAI format (messages).
    """
    system = ""
    user = ""

    if "conversations" in record:
        for turn in record["conversations"]:
            if turn["from"] == "system":
                system = turn["value"]
            elif turn["from"] == "human":
                user = turn["value"]
                break
    elif "messages" in record:
        for msg in record["messages"]:
            if msg["role"] == "system":
                system = msg["content"]
            elif msg["role"] == "user":
                user = msg["content"]
                break
    return system, user


def _to_sharegpt(system: str, user: str, response: str) -> dict:
    conversations = []
    if system:
        conversations.append({"from": "system", "value": system})
    conversations.append({"from": "human", "value": user})
    conversations.append({"from": "gpt", "value": response})
    return {"conversations": conversations}


def main() -> None:
    cfg = _config.load()

    parser = argparse.ArgumentParser(description="Distill from a teacher model")
    parser.add_argument("--teacher", choices=["anthropic", "vllm", "openai"],
                        default="anthropic")
    parser.add_argument("--teacher-model", type=str, default=None,
                        help="Teacher model name (default: claude-sonnet-4-20250514 for anthropic)")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None,
                        help="Max records to process (for testing)")
    parser.add_argument("--input", type=str, default=None,
                        help="Input JSONL (default: train_data from config)")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--teacher-url", type=str, default=None,
                        help="Base URL for vllm/openai teacher")
    args = parser.parse_args()

    INPUT = Path(args.input or str(cfg.train_data)).expanduser()
    OUTPUT = Path(args.output or os.environ.get(
        "DISTILL_OUT", str(cfg.data_dir / "distilled.jsonl")))

    # Set up teacher client
    if args.teacher == "anthropic":
        from anthropic import Anthropic
        teacher_client = Anthropic()
        teacher_model = args.teacher_model or "claude-sonnet-4-20250514"
        query_fn = _query_anthropic
    else:
        from openai import OpenAI
        if args.teacher == "vllm":
            base_url = args.teacher_url or os.environ.get("TEACHER_URL", cfg.vllm_url)
            teacher_client = OpenAI(base_url=f"{base_url}/v1", api_key="none")
            teacher_model = args.teacher_model or cfg.vllm_model
        else:
            base_url = args.teacher_url or os.environ.get("TEACHER_URL", "https://api.openai.com")
            teacher_client = OpenAI(base_url=f"{base_url}/v1")
            teacher_model = args.teacher_model or "gpt-4o"
        query_fn = _query_openai

    with open(INPUT) as fh:
        records = [json.loads(line) for line in fh]
    if args.limit:
        records = records[:args.limit]

    print(f"Teacher:     {args.teacher} ({teacher_model})")
    print(f"Input:       {INPUT} ({len(records)} records)")
    print(f"Max tokens:  {args.max_tokens}")
    print(f"Workers:     {args.workers}")
    print("=" * 60)

    results = []
    errors = 0

    def _process_one(idx: int, record: dict) -> tuple[int, dict | None]:
        system, user = _extract_prompt_parts(record)
        if not user:
            return idx, None
        try:
            response = query_fn(teacher_client, teacher_model, system, user,
                                args.max_tokens)
            return idx, _to_sharegpt(system, user, response)
        except Exception as e:
            print(f"  [ERROR] record {idx}: {type(e).__name__}: {e}", file=sys.stderr)
            return idx, None

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process_one, i, r): i for i, r in enumerate(records)}
        for fut in as_completed(futures):
            idx, result = fut.result()
            if result:
                results.append((idx, result))
                if len(results) % 10 == 0:
                    print(f"  Completed {len(results)}/{len(records)}")
            else:
                errors += 1

    # Sort by original order and write
    results.sort(key=lambda x: x[0])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as fh:
        for _, record in results:
            fh.write(json.dumps(record) + "\n")

    print("=" * 60)
    print(f"Processed: {len(records)}")
    print(f"Succeeded: {len(results)}")
    print(f"Errors:    {errors}")
    print(f"Output:    {OUTPUT}")

    meta_path = OUTPUT.with_suffix(".meta.json")
    meta = {
        "timestamp": datetime.now().isoformat(),
        "teacher": args.teacher,
        "teacher_model": teacher_model,
        "input_records": len(records),
        "output_records": len(results),
        "errors": errors,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
