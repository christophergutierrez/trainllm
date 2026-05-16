#!/usr/bin/env python3
"""
Evaluate a fine-tuned or base model against a holdout set and write a report.

Usage:
    python eval.py                                         # adapter from config.yaml
    MODEL=Qwen/Qwen2.5-Coder-14B-Instruct python eval.py  # base model
    HOLDOUT=~/data/holdout.jsonl python eval.py            # explicit holdout path
    VLLM_URL=http://localhost:8000 python eval.py          # override vLLM server URL

Holdout format (JSONL):
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
     "id": "optional-id", "conventions_tested": ["optional", "tags"]}

Output:
    <evals_dir>/<timestamp>_<model>.md    human-readable report
    <evals_dir>/<timestamp>_<model>.json  raw data
"""

import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import _config
from _eval_utils import THRESHOLDS, band, similarity, diagnostics


def query(client, model: str, record: dict) -> tuple[str, str, float, dict]:
    messages = [{"role": m["role"], "content": m["content"]}
                for m in record["messages"] if m["role"] != "assistant"]
    expected = next(
        (m["content"] for m in record["messages"] if m["role"] == "assistant"),
        "",
    )
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=800,
        temperature=0.0,  # deterministic greedy decoding — eliminates vLLM sampling variance
        seed=42,
    )
    choice = resp.choices[0]
    generated = choice.message.content or ""
    truncated = choice.finish_reason == "length"
    diag = diagnostics(generated, expected)
    diag["truncated"] = truncated
    diag["completion_tokens"] = resp.usage.completion_tokens if resp.usage else 0
    return generated, expected, similarity(expected, generated), diag


def _eval_one(client, model: str, i: int, r: dict) -> dict:
    rid = r.get("id", str(i))
    label = r.get("label", rid)
    try:
        generated, expected, score, diag = query(client, model, r)
        b = band(score)
        return {
            "id":                 rid,
            "label":              label,
            "source_file":        r.get("source_file", ""),
            "conventions_tested": r.get("conventions_tested", []),
            "score":              round(score, 4),
            "band":               b,
            "prompt":             next((m["content"] for m in r["messages"] if m["role"] == "user"), ""),
            "expected":           expected,
            "generated":          generated,
            "error":              None,
            **diag,
        }
    except Exception as e:
        return {
            "id": rid, "label": label,
            "conventions_tested": r.get("conventions_tested", []),
            "score": 0.0, "band": "ERROR",
            "prompt": "", "expected": "", "generated": "", "error": str(e),
            "length_ratio": 0.0,
        }


def main() -> None:
    from openai import OpenAI

    cfg = _config.load()

    BASE_URL  = os.environ.get("VLLM_URL",  cfg.vllm_url)
    MODEL     = os.environ.get("MODEL",     cfg.adapter_name)
    DATA      = Path(os.environ.get("HOLDOUT", str(cfg.holdout))).expanduser()
    EVALS_DIR = cfg.evals_dir
    EVALS_DIR.mkdir(parents=True, exist_ok=True)

    client = OpenAI(base_url=f"{BASE_URL}/v1", api_key="none")

    # ── Run evaluation ─────────────────────────────────────────────────────────

    with open(DATA) as fh:
        records = [json.loads(line) for line in fh]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    safe_model = MODEL.replace("/", "_")

    print(f"Model:   {MODEL}")
    print(f"Records: {len(records)}")
    print(f"Server:  {BASE_URL}")
    print("=" * 60)

    EVAL_WORKERS = int(os.environ.get("EVAL_WORKERS", "8"))

    print(f"Workers: {EVAL_WORKERS}")
    results = [None] * len(records)
    with ThreadPoolExecutor(max_workers=EVAL_WORKERS) as pool:
        futures = {pool.submit(_eval_one, client, MODEL, i, r): i for i, r in enumerate(records)}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                result = fut.result()
            except Exception as e:
                result = {
                    "id": records[idx].get("id", str(idx)),
                    "label": records[idx].get("label", str(idx)),
                    "conventions_tested": records[idx].get("conventions_tested", []),
                    "score": 0.0, "band": "ERROR",
                    "prompt": "", "expected": "", "generated": "",
                    "error": f"future_error: {e}", "length_ratio": 0.0,
                }
            results[idx] = result
            trunc_flag = " TRUNCATED" if result.get("truncated") else ""
            if result["band"] == "ERROR":
                print(f"  [ERROR     ]  {result['id']}  {result['label']}  {result['error']}")
            else:
                print(f"  [{result['band']:9s}  {result['score']:.2f}]  {result['id']}  {result['label']}{trunc_flag}")

    # ── Aggregate ──────────────────────────────────────────────────────────────

    scores      = [r["score"] for r in results if r["band"] != "ERROR"]
    json_scores = [r["json_score"] for r in results if r["band"] != "ERROR" and "json_score" in r]
    avg         = sum(scores) / len(scores) if scores else 0.0
    json_avg    = sum(json_scores) / len(json_scores) if json_scores else 0.0
    band_counts: defaultdict[str, int] = defaultdict(int)
    for r in results:
        band_counts[r["band"]] += 1

    conv_scores = defaultdict(list)
    for r in results:
        for c in r["conventions_tested"]:
            conv_scores[c].append(r["score"])

    conv_summary = sorted(
        [{"convention": c, "avg": round(sum(v) / len(v), 3), "n": len(v), "scores": v}
         for c, v in conv_scores.items()],
        key=lambda x: x["avg"],
    )

    poor_results     = [r for r in results if r["band"] in ("POOR", "PARTIAL", "ERROR")]
    weak_conventions = [c for c in conv_summary if c["avg"] < THRESHOLDS["good"]]
    truncated_count  = sum(1 for r in results if r.get("truncated"))
    if truncated_count:
        print(f"\n⚠ {truncated_count}/{len(results)} responses hit max_tokens (800) — check for verbosity issues")

    # ── Write JSON ─────────────────────────────────────────────────────────────

    json_path = EVALS_DIR / f"{timestamp}_{safe_model}.json"
    with open(json_path, "w") as f:
        json.dump({
            "meta": {
                "timestamp":    timestamp,
                "model":        MODEL,
                "base_url":     BASE_URL,
                "holdout_file": str(DATA),
                "n_records":    len(records),
                "base_model":   cfg.model,
                "lora_path":    str(cfg.final_dir) if MODEL == cfg.adapter_name else None,
            },
            "summary": {
                "avg_score":            round(avg, 4),
                "json_avg_score":       round(json_avg, 4),
                "band_counts":          dict(band_counts),
                "convention_breakdown": conv_summary,
                "weak_conventions":     weak_conventions,
            },
            "results": results,
        }, f, indent=2)

    # ── Write Markdown report ──────────────────────────────────────────────────

    md_path = EVALS_DIR / f"{timestamp}_{safe_model}.md"
    with open(md_path, "w") as f:
        def w(s: str = "") -> None:
            f.write(s + "\n")

        w(f"# Eval Report — {MODEL}")
        w(f"**Date:** {timestamp}  ")
        w(f"**Holdout:** `{DATA}`  ")
        w(f"**Base model:** `{cfg.model}`  ")
        if MODEL == cfg.adapter_name:
            w(f"**LoRA:** `{cfg.final_dir}`  ")
        w()

        w("## Overall Scores")
        w()
        w("| Metric | Value |")
        w("|--------|-------|")
        w(f"| Average similarity | {avg:.2f} |")
        w(f"| JSON-only avg      | {json_avg:.2f} |")
        w(f"| Excellent (≥0.8) | {band_counts['EXCELLENT']}/{len(results)} |")
        w(f"| Good (0.6–0.8)   | {band_counts['GOOD']}/{len(results)} |")
        w(f"| Partial (0.4–0.6)| {band_counts['PARTIAL']}/{len(results)} |")
        w(f"| Poor (<0.4)      | {band_counts['POOR']}/{len(results)} |")
        w(f"| Errors           | {band_counts['ERROR']}/{len(results)} |")
        w()

        if json_avg - avg > 0.03:
            w(f"> **Diagnostic:** JSON-only avg ({json_avg:.2f}) is significantly higher than "
              f"full-text avg ({avg:.2f}). This usually means the model's API calls are correct "
              f"but the thinking traces differ from holdout expectations (format mismatch, not a "
              f"capability issue).")
            w()

        if conv_summary:
            w("## Convention Breakdown")
            w()
            w("Sorted worst-first. Conventions below 0.6 need more training data.")
            w()
            w("| Convention | Avg Score | N | Band |")
            w("|------------|-----------|---|------|")
            for c in conv_summary:
                w(f"| `{c['convention']}` | {c['avg']:.2f} | {c['n']} | {band(c['avg'])} |")
            w()

        if weak_conventions:
            w("## Weak Conventions (priority targets for next training round)")
            w()
            for c in weak_conventions:
                w(f"- **`{c['convention']}`** — avg {c['avg']:.2f} across {c['n']} examples")
            w()
            w("These conventions need more training data before retraining.")
            w()

        w("## Per-Example Results")
        w()
        w("| ID | Label | Score | Band | Length Ratio |")
        w("|----|-------|-------|------|--------------|")
        for r in sorted(results, key=lambda x: x["score"]):
            w(f"| `{r['id']}` | `{r['label']}` | {r['score']:.2f} | {r['band']} | {r.get('length_ratio', '?')} |")
        w()

        if poor_results:
            w("## Failing Examples (POOR / PARTIAL / ERROR)")
            w()
            for r in poor_results:
                w(f"### {r['id']} — `{r['label']}` (score: {r['score']:.2f})")
                w()
                if r["conventions_tested"]:
                    w(f"**Conventions tested:** {', '.join(f'`{c}`' for c in r['conventions_tested'])}")
                    w()
                if r.get("source_file"):
                    w(f"**Source file:** `{r['source_file']}`")
                    w()
                if r["error"]:
                    w(f"**Error:** `{r['error']}`")
                else:
                    w("**Prompt:**")
                    w("```")
                    w(r["prompt"])
                    w("```")
                    w()
                    w("**Expected:**")
                    w("```")
                    w(r["expected"])
                    w("```")
                    w()
                    w("**Generated:**")
                    w("```")
                    w(r["generated"])
                    w("```")
                w()

        w("## Suggested Next Steps")
        w()
        w("_Based on these results, for a future training round:_")
        w()
        if not weak_conventions and avg >= 0.8:
            w("- Scores are strong. Consider evaluating on real tasks rather than synthetic holdout.")
            w("- Consider increasing model capacity: raise LoRA rank or switch to a larger base model.")
        elif not weak_conventions and avg >= 0.6:
            w("- Scores are acceptable. Check PARTIAL examples manually — similarity may undercount correct outputs.")
            w("- Focus next data generation on the lowest-scoring individual examples above.")
        else:
            w("Priority actions (in order):")
            w()
            for i, c in enumerate(weak_conventions[:5], 1):
                w(f"{i}. Generate more examples targeting `{c['convention']}` (currently {c['avg']:.2f})")
            w()
            w("General guidance:")
            w("- Do not increase training steps until data gaps are filled — more steps on thin data causes overfitting.")
            w("- After adding data, retrain from scratch for cleanest signal.")
            w("- Compare base model scores on the same holdout to confirm the issue is data, not model capacity.")
        w()
        w("---")
        w(f"_Raw data: `{json_path}`_")

    print()
    print("=" * 60)
    print(f"Avg similarity: {avg:.2f}  (JSON-only: {json_avg:.2f})")
    print(f"Excellent: {band_counts['EXCELLENT']}  Good: {band_counts['GOOD']}  "
          f"Partial: {band_counts['PARTIAL']}  Poor: {band_counts['POOR']}")
    if json_avg - avg > 0.03:
        print(f"\n⚠ JSON avg ({json_avg:.2f}) >> full avg ({avg:.2f}) — likely thinking-trace format mismatch, not a capability issue")
    print()
    print(f"Report: {md_path}")
    print(f"JSON:   {json_path}")


if __name__ == "__main__":
    main()
