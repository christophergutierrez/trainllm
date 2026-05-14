#!/usr/bin/env python3
"""
Prepare API training data for trainLLM.

Reads training.jsonl files from endpoint subdirectories, converts to ShareGPT
format, applies a system prompt, and produces stratified train/holdout splits.

Input record format:
    {"question": "List 10 resources",
     "api_call": {"endpoint": "GET /...", "params": {"pageSize": 10}}}

Output record format — training (ShareGPT):
    {"conversations": [
        {"from": "system", "value": "..."},
        {"from": "human",  "value": "List 10 resources"},
        {"from": "gpt",    "value": "```json\n{...}\n```"}
    ]}

Output record format — holdout (OpenAI messages):
    {"id": "resources-0042",
     "label": "List 10 resources",
     "messages": [
         {"role": "system",    "content": "..."},
         {"role": "user",      "content": "List 10 resources"},
         {"role": "assistant", "content": "```json\n{...}\n```"}
     ],
     "conventions_tested": ["resources", "list-endpoint", "page-size"]}

Usage:
    python prepare_data.py \\
        --input-dir ~/git_home/source_data/acme \\
        --train-out  ~/trainLLM/data/training.jsonl \\
        --holdout-out ~/trainLLM/data/holdout.jsonl

    # dry run to see counts without writing:
    python prepare_data.py --input-dir ... --dry-run
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

DEFAULT_ORG = os.environ.get("TRAINLLM_ORG", "acme")


def build_system_prompt(org_name: str, style: str = "conversational") -> str:
    org_label = org_name.strip() or DEFAULT_ORG
    if style == "structural":
        # Shorter, denser prompt for larger models (27B+).
        # Strips conversational filler; keeps VideoAmp as a latent-space anchor
        # and retains only functional format constraints.
        return (
            f"{org_label} API. Plan reasoning in <think> tags. Output: JSON code block.\n"
            "Single: {\"endpoint\": \"GET /...\", \"params\": {...}}\n"
            "Two-step: {\"steps\": [{\"endpoint\": \"GET /...\", \"params\": {}}, "
            "{\"endpoint\": \"GET /.../{id}\", \"params\": {\"id\": \"{{steps.0.fieldName}}\"}}]}"
        )
    # conversational (default) — better grounding for smaller models
    return (
        f"You are a {org_label} API assistant. "
        "Given a natural language request, respond with the correct API call "
        "as a JSON object inside a code block. "
        "Think through the request before answering. "
        "For a single call use: {\"endpoint\": \"GET /...\", \"params\": {...}}. "
        "For a two-step call (when an ID must be fetched first) use: "
        "{\"steps\": [{\"endpoint\": \"GET /...\", \"params\": {}}, "
        "{\"endpoint\": \"GET /.../{id}\", \"params\": {\"id\": \"{{steps.0.fieldName}}\"}}]}."
    )


def format_response(api_call: dict, thinking: str = None) -> str:
    """Render api_call as a fenced JSON code block, optionally preceded by a thinking block."""
    body = "```json\n" + json.dumps(api_call, indent=2) + "\n```"
    if thinking:
        return f"<think>\n{thinking}\n</think>\n{body}"
    return body


def to_sharegpt(record: dict, system_prompt: str) -> dict:
    return {
        "conversations": [
            {"from": "system", "value": system_prompt},
            {"from": "human",  "value": record["question"]},
            {"from": "gpt",    "value": format_response(record["api_call"], record.get("thinking"))},
        ]
    }


def to_holdout(record: dict, endpoint_name: str, idx: int, system_prompt: str) -> dict:
    response = format_response(record["api_call"], record.get("thinking"))
    params = record["api_call"].get("params", {})

    # Tag conventions: endpoint name + structural categories
    conventions = [endpoint_name]
    if "pageToken" in params:
        conventions.append("pagination")
    if "pageSize" in params and len(params) == 1:
        conventions.append("page-size-only")
    elif len(params) > 1:
        conventions.append("filtered")
    if not params:
        conventions.append("no-params")
    # Detect path-param endpoints (endpoint name is singular resource)
    if any(k not in ("pageSize", "pageToken") and not k.endswith("Id")
           for k in params):
        conventions.append("path-param")

    return {
        "id": f"{endpoint_name}-{idx:04d}",
        "label": record["question"],
        "messages": [
            {"role": "system",    "content": system_prompt},
            {"role": "user",      "content": record["question"]},
            {"role": "assistant", "content": response},
        ],
        "conventions_tested": conventions,
    }


def load_endpoint(path: Path) -> tuple[str, list[dict]]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return path.parent.name, records


def stratified_split(
    records_by_endpoint: dict[str, list[dict]],
    holdout_frac: float,
    seed: int,
) -> tuple[list[tuple[str, dict]], list[tuple[str, dict]]]:
    """
    Split per-endpoint, keeping at least 1 holdout record per endpoint
    regardless of size. Returns (train_items, holdout_items) where each
    item is (endpoint_name, record).
    """
    rng = random.Random(seed)
    train_items: list[tuple[str, dict]] = []
    holdout_items: list[tuple[str, dict]] = []

    for ep, records in sorted(records_by_endpoint.items()):
        shuffled = records[:]
        rng.shuffle(shuffled)
        n_holdout = max(1, round(len(shuffled) * holdout_frac))
        holdout_items.extend((ep, r) for r in shuffled[:n_holdout])
        train_items.extend((ep, r) for r in shuffled[n_holdout:])

    rng.shuffle(train_items)
    rng.shuffle(holdout_items)
    return train_items, holdout_items


def main():
    parser = argparse.ArgumentParser(description="Prepare API training data for trainLLM.")
    parser.add_argument("--input-dir", required=True,
                        help="Directory containing endpoint subdirs with training.jsonl files")
    parser.add_argument("--train-out", default=None,
                        help="Output path for training JSONL (ShareGPT format)")
    parser.add_argument("--holdout-out", default=None,
                        help="Output path for holdout JSONL (messages format)")
    parser.add_argument("--holdout-frac", type=float, default=0.10,
                        help="Fraction of each endpoint's records to use as holdout (default: 0.10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible splits (default: 42)")
    parser.add_argument("--org-name", default=DEFAULT_ORG,
                        help=f"Organization label for the generated system prompt (default: {DEFAULT_ORG})")
    parser.add_argument("--prompt-style", choices=["conversational", "structural"], default=None,
                        help="System prompt style: 'conversational' (default, better for <=8B) or "
                             "'structural' (~60%% shorter, better for 27B+). Overrides config trace_style.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print counts without writing any files")
    args = parser.parse_args()

    # Resolve prompt style: CLI flag > config field > default (conversational)
    prompt_style = args.prompt_style
    if prompt_style is None:
        # Could read from config here if passed; default to conversational
        prompt_style = os.environ.get("TRAINLLM_PROMPT_STYLE", "conversational")

    system_prompt = build_system_prompt(args.org_name, style=prompt_style)
    print(f"  Prompt style: {prompt_style}")

    input_dir = Path(args.input_dir).expanduser()
    if not input_dir.is_dir():
        sys.exit(f"Input directory not found: {input_dir}")

    # Discover all training.jsonl files
    jsonl_files = sorted(input_dir.glob("*/training.jsonl"))
    if not jsonl_files:
        sys.exit(f"No */training.jsonl files found under {input_dir}")

    records_by_endpoint: dict[str, list[dict]] = {}
    for path in jsonl_files:
        ep_name, records = load_endpoint(path)
        if records:
            records_by_endpoint[ep_name] = records
            print(f"  {ep_name:<30} {len(records):>4} records")

    total = sum(len(v) for v in records_by_endpoint.values())
    print(f"\n  Total: {total} records across {len(records_by_endpoint)} endpoints")

    train_items, holdout_items = stratified_split(
        records_by_endpoint, args.holdout_frac, args.seed
    )

    print(f"  Split:  {len(train_items)} train  |  {len(holdout_items)} holdout "
          f"({args.holdout_frac:.0%} holdout fraction)")

    if args.dry_run:
        print("\nDry run — no files written.")
        return

    if not args.train_out and not args.holdout_out:
        sys.exit("Specify --train-out and/or --holdout-out, or use --dry-run.")

    if args.train_out:
        out = Path(args.train_out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for _, record in train_items:
                f.write(json.dumps(to_sharegpt(record, system_prompt)) + "\n")
        print(f"\n  Training:  {len(train_items)} records → {out}")

    if args.holdout_out:
        out = Path(args.holdout_out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)

        # Preserve any hand-curated records (canonical-*, mcp-*) from a prior run
        preserved = []
        if out.exists():
            for line in out.read_text().splitlines():
                if not line.strip(): continue
                r = json.loads(line)
                if any(r.get("id","").startswith(p) for p in ("canonical-","mcp-","mcp_")):
                    preserved.append(line)

        ep_counters: dict[str, int] = {}
        with open(out, "w") as f:
            # Write hand-curated records first so they survive future regenerations
            for line in preserved:
                f.write(line + "\n")
            for ep, record in holdout_items:
                idx = ep_counters.get(ep, 0)
                ep_counters[ep] = idx + 1
                f.write(json.dumps(to_holdout(record, ep, idx, system_prompt)) + "\n")
        kept = len(preserved)
        print(f"  Holdout:   {len(holdout_items)} generated + {kept} preserved → {len(holdout_items)+kept} total → {out}")


if __name__ == "__main__":
    main()
