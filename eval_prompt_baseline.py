#!/usr/bin/env python3
"""
One-off: evaluate the base model with a conventions system-prompt injected,
for head-to-head comparison against the cycle-12 adapter.

Input:
    handoff/prompt_engineered_baseline.yaml  — reposynth's system prompt
    data/holdout.jsonl                        — same 30 records used by cycle 12

Output:
    evals/prompt_baseline/<ts>_prompt_baseline_Qwen_Qwen2.5-Coder-14B-Instruct.{md,json}
"""

import json
import sys
from collections import defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
import re

import yaml  # type: ignore[import-untyped]
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
import _config
cfg = _config.load()

BASE_URL  = cfg.vllm_url
MODEL     = cfg.model                                 # Qwen/Qwen2.5-Coder-14B-Instruct, no adapter
DATA      = cfg.holdout
PROMPT_YAML = Path(__file__).parent / "handoff" / "prompt_engineered_baseline.yaml"
OUT_DIR   = cfg.evals_dir / "prompt_baseline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = yaml.safe_load(PROMPT_YAML.read_text())["system_prompt"]
print(f"System prompt: {len(SYSTEM_PROMPT)} chars, ~{len(SYSTEM_PROMPT)//4} tokens estimated")

client = OpenAI(base_url=f"{BASE_URL}/v1", api_key="none")

THRESHOLDS = {"excellent": 0.8, "good": 0.6, "partial": 0.4}


def band(score: float) -> str:
    if score >= THRESHOLDS["excellent"]: return "EXCELLENT"
    if score >= THRESHOLDS["good"]:      return "GOOD"
    if score >= THRESHOLDS["partial"]:   return "PARTIAL"
    return "POOR"


def _strip_fences(text: str) -> str:
    text = re.sub(r"^```[\w]*\n?", "", text.strip())
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.strip(), b.strip()).ratio()


def _diagnostics(generated: str, expected: str) -> dict:
    gen = _strip_fences(generated)
    exp = _strip_fences(expected)
    return {"length_ratio": round(len(gen) / max(len(exp), 1), 2)}


def query(record: dict) -> tuple[str, str, float, dict]:
    """Build messages with reposynth's system prompt replacing the holdout's existing one."""
    user_turns = [m for m in record["messages"] if m["role"] == "user"]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + \
               [{"role": "user", "content": m["content"]} for m in user_turns]
    expected = next(m["content"] for m in record["messages"] if m["role"] == "assistant")
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=1024,
        temperature=0.0,
        seed=42,
    )
    generated = resp.choices[0].message.content or ""
    diag = _diagnostics(generated, expected)
    return generated, expected, similarity(expected, generated), diag


# ── Run ──

records = [json.loads(l) for l in open(DATA)]
timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
safe_model = MODEL.replace("/", "_")

print(f"Model:   {MODEL} (base, no adapter) + prompt-engineered system message")
print(f"Records: {len(records)}")
print(f"Server:  {BASE_URL}")
print("=" * 60)

results = []
for i, r in enumerate(records):
    rid = r.get("id", str(i))
    label = r.get("label", rid)
    try:
        generated, expected, score, diag = query(r)
        b = band(score)
        print(f"  [{b:9s}  {score:.2f}]  {rid}  {label}")
        results.append({
            "id":                 rid,
            "label":              label,
            "source_file":        r.get("source_file", ""),
            "conventions_tested": r.get("conventions_tested", []),
            "score":              round(score, 4),
            "band":               b,
            "prompt":             next(m["content"] for m in r["messages"] if m["role"] == "user"),
            "expected":           expected,
            "generated":          generated,
            "error":              None,
            **diag,
        })
    except Exception as e:
        print(f"  [ERROR     ]  {rid}  {label}  {e}")
        results.append({
            "id": rid, "label": label,
            "conventions_tested": r.get("conventions_tested", []),
            "score": 0.0, "band": "ERROR",
            "prompt": "", "expected": "", "generated": "", "error": str(e),
            "length_ratio": 0.0,
        })

# ── Aggregate ──

scores = [r["score"] for r in results if r["band"] != "ERROR"]
avg = sum(scores) / len(scores) if scores else 0.0
band_counts: dict = defaultdict(int)
for r in results:
    band_counts[r["band"]] += 1

conv_scores: dict = defaultdict(list)
for r in results:
    for c in r["conventions_tested"]:
        conv_scores[c].append(r["score"])
conv_summary = sorted(
    [{"convention": c, "avg": round(sum(v) / len(v), 3), "n": len(v), "scores": v}
     for c, v in conv_scores.items()],
    key=lambda x: x["avg"],
)

# ── Write JSON ──

json_path = OUT_DIR / f"{timestamp}_prompt_baseline_{safe_model}.json"
with open(json_path, "w") as f:
    json.dump({
        "meta": {
            "timestamp":    timestamp,
            "model":        f"{MODEL} + prompt_engineered_baseline system prompt",
            "base_url":     BASE_URL,
            "holdout_file": str(DATA),
            "n_records":    len(records),
            "base_model":   MODEL,
            "lora_path":    None,
            "system_prompt_source": str(PROMPT_YAML),
            "system_prompt_chars":  len(SYSTEM_PROMPT),
        },
        "summary": {
            "avg_score":            round(avg, 4),
            "band_counts":          dict(band_counts),
            "convention_breakdown": conv_summary,
        },
        "results": results,
    }, f, indent=2)

# ── Short markdown ──

md_path = OUT_DIR / f"{timestamp}_prompt_baseline_{safe_model}.md"
with open(md_path, "w") as f:
    f.write(f"# Prompt-Engineered Baseline Eval\n\n")
    f.write(f"**Date:** {timestamp}  \n")
    f.write(f"**Model:** `{MODEL}` (base, no adapter)  \n")
    f.write(f"**System prompt source:** `{PROMPT_YAML.name}` ({len(SYSTEM_PROMPT)} chars)  \n")
    f.write(f"**Holdout:** `{DATA}` ({len(records)} records)  \n\n")
    f.write(f"## Overall\n\n")
    f.write(f"| Metric | Value |\n|---|---|\n")
    f.write(f"| Average similarity | {avg:.3f} |\n")
    f.write(f"| Excellent (≥0.8) | {band_counts['EXCELLENT']}/{len(results)} |\n")
    f.write(f"| Good (0.6–0.8) | {band_counts['GOOD']}/{len(results)} |\n")
    f.write(f"| Partial (0.4–0.6) | {band_counts['PARTIAL']}/{len(results)} |\n")
    f.write(f"| Poor (<0.4) | {band_counts['POOR']}/{len(results)} |\n\n")
    f.write(f"## Convention breakdown\n\n| convention | avg | n |\n|---|---|---|\n")
    for c in conv_summary:
        f.write(f"| `{c['convention']}` | {c['avg']:.3f} | {c['n']} |\n")

print()
print("=" * 60)
print(f"Avg similarity: {avg:.3f}")
print(f"Excellent: {band_counts['EXCELLENT']}  Good: {band_counts['GOOD']}  "
      f"Partial: {band_counts['PARTIAL']}  Poor: {band_counts['POOR']}")
print()
print(f"Report: {md_path}")
print(f"JSON:   {json_path}")
