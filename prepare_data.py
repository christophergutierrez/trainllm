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
import re
import sys
from pathlib import Path

DEFAULT_ORG = os.environ.get("TRAINLLM_ORG", "acme")


# ── QOC trace converter ──────────────────────────────────────────────────────

def _parse_trace(thinking: str) -> dict:
    """Parse a linear thinking trace into structured fields."""
    lines = [l.strip() for l in thinking.strip().split("\n") if l.strip()]
    kv: dict[str, str] = {}
    nots: list[str] = []
    extras: list[str] = []

    KEYS = [
        "Requested count", "Two-step chain", "Possession note", "Possession",
        "No filter", "Disambiguation", "Descriptor", "Entity", "Scope",
        "Domain", "Endpoint", "Filters", "Params", "Goal", "Step 0",
        "Step 1", "Note",
    ]

    for line in lines:
        if line.startswith("NOT:"):
            nots.append(line[4:].strip())
            continue
        if line.startswith("Use:"):
            kv["use"] = line[4:].strip()
            continue
        matched = False
        for key in KEYS:
            if line.startswith(key + ":"):
                val = line[len(key) + 1 :].strip()
                norm = key.lower().replace(" ", "_")
                if norm not in kv:
                    kv[norm] = val
                else:
                    extras.append(val)
                matched = True
                break
        if not matched:
            extras.append(line)

    return {**kv, "nots": nots, "extras": extras}


def _split_endpoint(s: str) -> tuple[str, str]:
    """Split 'GET /path (reason text)' → (endpoint, reason)."""
    m = re.match(r"(GET \S+)\s*(?:\((.+)\))?", s)
    if m:
        return m.group(1), m.group(2) or ""
    return s, ""


def _criteria(kv: dict, extras: list[str], skip: set[str] | None = None) -> list[str]:
    """Gather criteria fragments from parsed fields."""
    skip = skip or set()
    parts: list[str] = []
    for key in ("scope", "requested_count", "filters", "possession",
                "possession_note", "no_filter", "descriptor", "disambiguation"):
        if key in skip:
            continue
        v = kv.get(key)
        if v:
            if key == "requested_count":
                parts.append(f"count={v}")
            elif key == "filters":
                parts.append(f"filter: {v}")
            else:
                parts.append(v.rstrip("."))
    parts.extend(e.rstrip(".") for e in extras)
    return parts


def _qoc_simple(kv: dict) -> str:
    endpoint = kv.get("endpoint") or kv.get("use", "?")
    entity = kv.get("entity", "resource")
    c = _criteria(kv, kv["extras"])
    out = [f"Question: Query {entity}?", f"Option: {endpoint}"]
    if c:
        out.append(f"Criteria: {'. '.join(c)}.")
    if kv.get("params"):
        out.append(f"Params: {kv['params']}")
    return "\n".join(out)


def _qoc_byid(kv: dict) -> str:
    use_ep, use_note = _split_endpoint(kv.get("use", "?"))
    entity = kv.get("entity", "resource")
    c = _criteria(kv, kv["extras"])
    out = [f"Question: Retrieve one {entity} or list?", f"Option A: {use_ep}"]
    for i, not_str in enumerate(kv["nots"]):
        ep, reason = _split_endpoint(not_str)
        label = chr(66 + i)
        out.append(f"Option {label}: {ep}")
        if reason:
            c.append(f"Option {label} rejected — {reason.rstrip('.')}")
    if c:
        out.append(f"Criteria: {'. '.join(c)}.")
    if kv.get("params"):
        out.append(f"Params: {kv['params']}")
    return "\n".join(out)


def _qoc_synonym(kv: dict) -> str:
    endpoint = kv.get("endpoint") or kv.get("use", "?")
    ep_clean, _ = _split_endpoint(endpoint)
    entity = kv.get("entity", "resource")
    c = _criteria(kv, kv["extras"])
    if kv.get("domain"):
        c.insert(0, kv["domain"].rstrip("."))
    out = [f'Question: Which endpoint for "{entity}"?']
    if kv["nots"]:
        out.append(f"Option A: {ep_clean}")
        for i, not_str in enumerate(kv["nots"]):
            ep, reason = _split_endpoint(not_str)
            label = chr(66 + i)
            out.append(f"Option {label}: {ep}")
            if reason:
                c.append(f"Option {label} rejected — {reason.rstrip('.')}")
            else:
                c.append(f"Option {label} rejected")
    else:
        out.append(f"Option: {ep_clean}")
    if c:
        out.append(f"Criteria: {'. '.join(c)}.")
    if kv.get("params"):
        out.append(f"Params: {kv['params']}")
    return "\n".join(out)


def _qoc_chain(kv: dict) -> str:
    step0 = kv.get("step_0", "")
    step1 = kv.get("step_1", "")
    s0_m = re.search(r"(GET \S+)", step0)
    s1_m = re.search(r"(GET \S+)", step1)
    s0_ep = s0_m.group(1) if s0_m else step0
    s1_ep = s1_m.group(1) if s1_m else step1
    # Extract the linking field from step1 (e.g. {{steps.0.audienceId}})
    field_m = re.search(r"\{\{(steps\.0\.\w+)\}\}", step1)
    field = field_m.group(1) if field_m else "steps.0.id"
    c = ["No ID provided", "Must resolve from list", "Option A rejected"]
    c.extend(e.rstrip(".") for e in kv["extras"])
    out = [
        "Question: Direct call or chain?",
        f"Option A: {s1_ep} — needs ID",
        f"Option B: Chain — {s0_ep} → {s1_ep}",
        f"Criteria: {'. '.join(c)}.",
        f"Params: chain via {{{{{field}}}}}",
    ]
    return "\n".join(out)


def convert_to_qoc(thinking: str) -> str:
    """Convert a linear thinking trace to QOC (Question-Option-Criteria) format."""
    if not thinking:
        return thinking
    kv = _parse_trace(thinking)
    if "goal" in kv or "step_0" in kv:
        return _qoc_chain(kv)
    if "domain" in kv:
        return _qoc_synonym(kv)
    if kv["nots"]:
        return _qoc_byid(kv)
    return _qoc_simple(kv)


def _parse_qoc(thinking: str) -> dict:
    """Parse a QOC thinking trace into structured fields."""
    lines = [l.strip() for l in thinking.strip().split("\n") if l.strip()]
    question = ""
    options: list[tuple[str, str]] = []  # (label_or_empty, endpoint_text)
    criteria = ""
    params = ""

    for line in lines:
        if line.startswith("Question:"):
            question = line[len("Question:"):].strip()
        elif re.match(r"Option\s*[A-Z]?:", line):
            m = re.match(r"Option\s*([A-Z])?:\s*(.*)", line)
            if m:
                options.append((m.group(1) or "", m.group(2).strip()))
        elif line.startswith("Criteria:"):
            criteria = line[len("Criteria:"):].strip()
        elif line.startswith("Params:"):
            params = line[len("Params:"):].strip()

    return {"question": question, "options": options, "criteria": criteria, "params": params}


def _linear_from_synonym(qoc: dict) -> str:
    m = re.search(r'"([^"]+)"', qoc["question"])
    entity = m.group(1) if m else "resource"
    out = [f"Entity: {entity}"]

    options = qoc["options"]
    use_ep = options[0][1] if options else "?"
    not_eps = options[1:]

    rejection = {}
    for rm in re.finditer(r"Option ([A-Z]) rejected\s*—?\s*([^.]*)", qoc["criteria"]):
        rejection[rm.group(1)] = rm.group(2).strip().rstrip(".")

    crit = re.sub(r"\s*Option [A-Z] rejected\s*—?\s*[^.]*\.?\s*", " ", qoc["criteria"]).strip()

    sentences = [s.strip() for s in crit.split(".") if s.strip()]
    domain_parts, scope_parts, extras = [], [], []
    found_scope = False
    for s in sentences:
        sl = s.lower()
        if not found_scope and ("list all" in sl or "single item" in sl):
            scope_parts.append(s)
            found_scope = True
        elif found_scope:
            extras.append(s)
        else:
            domain_parts.append(s)

    if domain_parts:
        out.append(f"Domain: {'. '.join(domain_parts)}.")
    for e in extras:
        el = e.lower()
        if "not a query parameter" in el or "do not add" in el:
            out.append(f"Descriptor: {e}.")
        elif "does not" in el and "filter" in el:
            out.append(f"No filter: {e}.")
        elif "'my" in el or "my " in el and "filter" in el:
            out.append(f"Possession note: {e}.")
        else:
            out.append(e)
    if scope_parts:
        out.append(f"Scope: {'. '.join(scope_parts)}")

    out.append(f"Use:    {use_ep}")
    for label, ep in not_eps:
        reason = rejection.get(label, "")
        out.append(f"NOT:    {ep} ({reason})" if reason else f"NOT:    {ep}")

    if qoc["params"]:
        out.append(f"Params: {qoc['params']}")
    return "\n".join(out)


def _linear_from_byid(qoc: dict) -> str:
    m = re.match(r"Retrieve one (\w+)", qoc["question"])
    entity = m.group(1) if m else "resource"
    out = [f"Entity: {entity}"]

    options = qoc["options"]
    use_ep = options[0][1] if options else "?"
    not_eps = options[1:]

    rejection = {}
    for rm in re.finditer(r"Option ([A-Z]) rejected\s*—?\s*([^.]*)", qoc["criteria"]):
        rejection[rm.group(1)] = rm.group(2).strip().rstrip(".")

    crit = re.sub(r"\s*Option [A-Z] rejected\s*—?\s*[^.]*\.?\s*", " ", qoc["criteria"]).strip()

    sentences = [s.strip() for s in crit.split(".") if s.strip()]
    if sentences:
        out.append(f"Scope: {sentences[0]}")
    for s in sentences[1:]:
        out.append(s)

    out.append(f"Use:    {use_ep}")
    for label, ep in not_eps:
        reason = rejection.get(label, "")
        out.append(f"NOT:    {ep} ({reason})" if reason else f"NOT:    {ep}")

    if qoc["params"]:
        out.append(f"Params: {qoc['params']}")
    return "\n".join(out)


def _linear_from_simple(qoc: dict) -> str:
    m = re.match(r"Query (.+?)\?", qoc["question"])
    entity = m.group(1) if m else "resource"
    out = [f"Entity: {entity}"]

    use_ep = qoc["options"][0][1] if qoc["options"] else "?"

    if qoc["criteria"]:
        crit = qoc["criteria"].rstrip(".")
        if ":" in crit.split(".")[0]:
            out.append(crit)
        else:
            out.append(f"Scope: {crit}")

    out.append(f"Endpoint: {use_ep}")

    if qoc["params"]:
        out.append(f"Params: {qoc['params']}")
    return "\n".join(out)


def _linear_from_chain(qoc: dict) -> str:
    options = qoc["options"]
    s1_ep = re.search(r"(GET \S+)", options[0][1]).group(1) if options else "?"
    chain_m = re.search(r"Chain\s*—?\s*(GET \S+)\s*→\s*(GET \S+)", options[1][1]) if len(options) > 1 else None
    s0_ep = chain_m.group(1) if chain_m else "?"
    field_m = re.search(r"\{\{(steps\.0\.\w+)\}\}", qoc["params"])
    field = field_m.group(1) if field_m else "steps.0.id"

    out = [
        f"Goal: get a single item but no ID was given",
        f"Two-step chain: yes",
        f"Step 0: {s0_ep} → scan list",
        f"Step 1: {s1_ep.replace('{', '{{').replace('}', '}}')} → use {{{{{field}}}}}",
    ]
    crit = re.sub(r"\s*Option [A-Z] rejected\s*—?\s*[^.]*\.?\s*", " ", qoc["criteria"]).strip()
    extras = [s.strip() for s in crit.split(".") if s.strip()
              and "no id provided" not in s.lower()
              and "must resolve" not in s.lower()]
    for e in extras:
        out.append(e)
    return "\n".join(out)


def convert_to_linear(thinking: str) -> str:
    """Convert a QOC thinking trace back to linear format."""
    if not thinking:
        return thinking
    if "Question:" not in thinking:
        return thinking
    qoc = _parse_qoc(thinking)
    q = qoc["question"]
    if "Direct call or chain" in q:
        return _linear_from_chain(qoc)
    if "Which endpoint" in q:
        return _linear_from_synonym(qoc)
    if "Retrieve one" in q:
        return _linear_from_byid(qoc)
    return _linear_from_simple(qoc)


def build_system_prompt(org_name: str, style: str = "conversational") -> str:
    org_label = org_name.strip() or DEFAULT_ORG
    if style == "structural":
        return (
            f"{org_label} API. Plan reasoning in <think> tags. Output: JSON code block.\n"
            "Single: {\"endpoint\": \"GET /...\", \"params\": {...}}\n"
            "Two-step: {\"steps\": [{\"endpoint\": \"GET /...\", \"params\": {}}, "
            "{\"endpoint\": \"GET /.../{id}\", \"params\": {\"id\": \"{{steps.0.fieldName}}\"}}]}"
        )
    if style == "qoc":
        return (
            f"{org_label} API assistant. "
            "Reason in <think> tags, then output a JSON code block. "
            "Single call: {\"endpoint\": \"GET /...\", \"params\": {...}}. "
            "Two-step (ID lookup first): "
            "{\"steps\": [{\"endpoint\": \"GET /...\", \"params\": {}}, "
            "{\"endpoint\": \"GET /.../{id}\", \"params\": {\"id\": \"{{steps.0.fieldName}}\"}}]}."
        )
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


def format_response(api_call: dict, thinking: str | None = None, trace_style: str = "linear") -> str:
    """Render api_call as a fenced JSON code block, optionally preceded by a thinking block."""
    body = "```json\n" + json.dumps(api_call, indent=2) + "\n```"
    if thinking:
        t = convert_to_qoc(thinking) if trace_style == "qoc" else thinking
        return f"<think>\n{t}\n</think>\n{body}"
    return body


def to_sharegpt(record: dict, system_prompt: str, trace_style: str = "linear") -> dict:
    return {
        "conversations": [
            {"from": "system", "value": system_prompt},
            {"from": "human",  "value": record["question"]},
            {"from": "gpt",    "value": format_response(record["api_call"], record.get("thinking"), trace_style)},
        ]
    }


def to_holdout(record: dict, endpoint_name: str, idx: int, system_prompt: str,
               trace_style: str = "linear") -> dict:
    response = format_response(record["api_call"], record.get("thinking"), trace_style)
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
                             "'structural' (~60%% shorter, better for 27B+). Overrides TRAINLLM_PROMPT_STYLE env var.")
    parser.add_argument("--trace-style", choices=["linear", "qoc"], default="linear",
                        help="Thinking trace format: 'linear' (Entity/Scope/Use/NOT) or "
                             "'qoc' (Question/Option/Criteria — forces explicit option rejection)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print counts without writing any files")
    args = parser.parse_args()

    trace_style = args.trace_style

    # Resolve prompt style: CLI flag > trace_style match > env > default
    prompt_style = args.prompt_style
    if prompt_style is None:
        if trace_style == "qoc":
            prompt_style = "qoc"
        else:
            prompt_style = os.environ.get("TRAINLLM_PROMPT_STYLE", "conversational")
    system_prompt = build_system_prompt(args.org_name, style=prompt_style)
    print(f"  Prompt style: {prompt_style}")
    print(f"  Trace style:  {trace_style}")

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
                f.write(json.dumps(to_sharegpt(record, system_prompt, trace_style)) + "\n")
        print(f"\n  Training:  {len(train_items)} records → {out}")

    if args.holdout_out:
        out = Path(args.holdout_out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)

        # Preserve any hand-curated records (canonical-*, mcp-*) from a prior run,
        # updating their system prompt and thinking traces to match current style
        preserved = []
        if out.exists():
            for line in out.read_text().splitlines():
                if not line.strip(): continue
                r = json.loads(line)
                if any(r.get("id","").startswith(p) for p in ("canonical-","mcp-","mcp_")):
                    for msg in r.get("messages", []):
                        if msg["role"] == "system":
                            msg["content"] = system_prompt
                        if msg["role"] == "assistant" and "<think>" in msg["content"]:
                            m = re.search(r"<think>\n(.*?)\n</think>", msg["content"], re.DOTALL)
                            if m:
                                old_think = m.group(1)
                                is_qoc = "Question:" in old_think and "Option" in old_think
                                if trace_style == "qoc" and not is_qoc:
                                    new_think = convert_to_qoc(old_think)
                                elif trace_style == "linear" and is_qoc:
                                    new_think = convert_to_linear(old_think)
                                else:
                                    new_think = None
                                if new_think is not None:
                                    msg["content"] = msg["content"].replace(
                                        f"<think>\n{old_think}\n</think>",
                                        f"<think>\n{new_think}\n</think>",
                                    )
                    preserved.append(json.dumps(r))

        ep_counters: dict[str, int] = {}
        tmp_out = out.with_suffix(".tmp")
        with open(tmp_out, "w") as f:
            for line in preserved:
                f.write(line + "\n")
            for ep, record in holdout_items:
                idx = ep_counters.get(ep, 0)
                ep_counters[ep] = idx + 1
                f.write(json.dumps(to_holdout(record, ep, idx, system_prompt, trace_style)) + "\n")
        os.replace(tmp_out, out)
        kept = len(preserved)
        total_holdout = len(holdout_items) + kept
        print(f"  Holdout:   {len(holdout_items)} generated + {kept} preserved → {total_holdout} total → {out}")

        # Format consistency check
        n_qoc = n_linear = n_other = 0
        for line in out.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            for msg in r.get("messages", []):
                if msg.get("role") == "assistant" and "<think>" in msg.get("content", ""):
                    if "Question:" in msg["content"] and "Option" in msg["content"]:
                        n_qoc += 1
                    elif "Entity:" in msg["content"] or "Scope:" in msg["content"]:
                        n_linear += 1
                    else:
                        n_other += 1
        if n_qoc > 0 and n_linear > 0:
            print(f"  ⚠ WARNING: Mixed trace formats in holdout — {n_qoc} QOC + {n_linear} linear + {n_other} other")
            print(f"    This will cause false regressions in eval. Fix with trace_style={trace_style}.")
        else:
            fmt = "QOC" if n_qoc > n_linear else "linear"
            print(f"  Trace format: {fmt} ({n_qoc + n_linear + n_other}/{total_holdout} records checked)")


if __name__ == "__main__":
    main()
