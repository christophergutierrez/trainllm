#!/usr/bin/env python3
"""
benchmark_demo.py — measure the three claims the eval report doesn't:
speed, tokens, and "the frontier can't do this."

Drops into the trainllm repo root (imports _eval_utils and _config). Assumes
the vLLM server from cycle.py is running and serving BOTH the base model and
the LoRA module (cycle.py's serve step already does this).

Arms:
  base     - base model, no adapter, via local vLLM
  adapter  - base + LoRA module, via local vLLM
  frontier - (optional, --frontier) an API model with NO tool access, same
             questions. It has never seen the fictional endpoints; the point
             is to show that institutional knowledge beats model size on
             this task. Requires ANTHROPIC_API_KEY.

Per request we record wall-clock latency, prompt/completion tokens, and the
same composite score eval.py uses — one definition of "correct" everywhere.

Usage:
  python benchmark_demo.py                       # base vs adapter, n=20
  python benchmark_demo.py --n 50
  python benchmark_demo.py --frontier            # adds the frontier arm
  python benchmark_demo.py --frontier --frontier-model claude-opus-4-8

Output: evals/benchmark_demo.json + evals/benchmark_demo.md
"""

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

from _config import load as load_config
from _eval_utils import similarity, structural_score, composite_score

MAX_TOKENS = 800
SEED = 42


def load_holdout(path: Path, n: int) -> list[dict]:
    records = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if not records:
        sys.exit(f"ERROR: no records in {path}")
    random.seed(SEED)
    return random.sample(records, min(n, len(records)))


def split_record(record: dict) -> tuple[list[dict], str]:
    """(messages without assistant turn, expected assistant content)."""
    msgs = [{"role": m["role"], "content": m["content"]}
            for m in record["messages"] if m["role"] != "assistant"]
    expected = next((m["content"] for m in record["messages"]
                     if m["role"] == "assistant"), "")
    return msgs, expected


def query_openai_compat(client, model: str, messages: list[dict]) -> dict:
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model, messages=messages,
        max_tokens=MAX_TOKENS, temperature=0.0, seed=SEED,
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    u = resp.usage
    return {
        "text": resp.choices[0].message.content or "",
        "latency_ms": latency_ms,
        "prompt_tokens": u.prompt_tokens if u else 0,
        "completion_tokens": u.completion_tokens if u else 0,
    }


def query_anthropic(client, model: str, messages: list[dict]) -> dict:
    system = "\n".join(m["content"] for m in messages if m["role"] == "system")
    user_msgs = [m for m in messages if m["role"] == "user"]
    t0 = time.perf_counter()
    resp = client.messages.create(
        model=model,
        system=system or None,
        messages=user_msgs,
        max_tokens=MAX_TOKENS,
        temperature=0.0,
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    return {
        "text": text,
        "latency_ms": latency_ms,
        "prompt_tokens": resp.usage.input_tokens,
        "completion_tokens": resp.usage.output_tokens,
    }


def run_arm(name: str, records: list[dict], ask, scoring) -> list[dict]:
    rows = []
    for i, r in enumerate(records):
        messages, expected = split_record(r)
        try:
            out = ask(messages)
        except Exception as e:  # noqa: BLE001 — record and continue
            rows.append({"id": r.get("id", str(i)), "error": str(e)})
            print(f"  [{name}] {i+1}/{len(records)} ERROR: {e}", file=sys.stderr)
            continue
        sim = similarity(expected, out["text"])
        struct = structural_score(expected, out["text"], scoring=scoring)
        rows.append({
            "id": r.get("id", str(i)),
            "conventions_tested": r.get("conventions_tested", []),
            "composite": composite_score(sim, struct, scoring=scoring),
            **{k: out[k] for k in
               ("latency_ms", "prompt_tokens", "completion_tokens")},
        })
        print(f"  [{name}] {i+1}/{len(records)} "
              f"score={rows[-1]['composite']:.3f} "
              f"{out['latency_ms']:.0f}ms {out['completion_tokens']}tok")
    return rows


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if "error" not in r]
    if not ok:
        return {"n": 0, "errors": len(rows)}
    med = lambda k: statistics.median(r[k] for r in ok)  # noqa: E731
    return {
        "n": len(ok),
        "errors": len(rows) - len(ok),
        "mean_composite": round(sum(r["composite"] for r in ok) / len(ok), 4),
        "median_latency_ms": round(med("latency_ms")),
        "median_prompt_tokens": int(med("prompt_tokens")),
        "median_completion_tokens": int(med("completion_tokens")),
    }


def write_report(path: Path, summaries: dict, n: int, frontier_model: str | None) -> None:
    s = summaries
    lines = [
        "# Benchmark: base vs adapter" + (" vs frontier" if "frontier" in s else ""),
        "",
        f"{n} holdout questions, temperature 0, max_tokens {MAX_TOKENS}. "
        "Same scorer as eval.py (composite of similarity + structural).",
        "",
        "| arm | n | mean composite | median latency (ms) | median completion tokens |",
        "|---|---|---|---|---|",
    ]
    for arm, v in s.items():
        if v.get("n", 0) == 0:
            lines.append(f"| {arm} | 0 ({v.get('errors', 0)} errors) | — | — | — |")
            continue
        lines.append(f"| {arm} | {v['n']} | {v['mean_composite']} "
                     f"| {v['median_latency_ms']} | {v['median_completion_tokens']} |")
    if "frontier" in s and s["frontier"].get("n"):
        lines += [
            "",
            f"Frontier arm: `{frontier_model}`, no tool access. A low frontier score "
            "on a high-scoring adapter task is the demo's closing argument: the "
            "pattern lives in the weights, and model size does not substitute for "
            "having seen the logs.",
        ]
    lines += [
        "",
        "Notes: latency compares a local vLLM round-trip against a remote API and "
        "includes network for the frontier arm — direction matters more than the "
        "exact ratio. Token columns are per-request medians; the cost claim follows "
        "from completion+prompt tokens at the respective prices.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=20, help="holdout sample size")
    p.add_argument("--holdout", default=None, help="override holdout path")
    p.add_argument("--base", default=None, help="override base model id")
    p.add_argument("--adapter", default=None, help="override LoRA module name")
    p.add_argument("--port", type=int, default=None, help="override vLLM port")
    p.add_argument("--frontier", action="store_true",
                   help="add frontier arm (needs ANTHROPIC_API_KEY)")
    p.add_argument("--frontier-model", default="claude-opus-4-8")
    args = p.parse_args()

    cfg = load_config()
    holdout = Path(args.holdout or cfg.holdout).expanduser()
    base = args.base or cfg.model
    adapter = args.adapter or cfg.adapter_name
    port = args.port or getattr(cfg, "vllm_port", 8000)
    scoring = getattr(cfg, "scoring", None)

    records = load_holdout(holdout, args.n)
    print(f"{len(records)} questions from {holdout}")
    print(f"vLLM :{port}  base={base}  adapter={adapter}")

    from openai import OpenAI
    local = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="EMPTY")

    summaries = {}
    rows_all = {}
    for arm, model in (("base", base), ("adapter", adapter)):
        print(f"\n== {arm} ==")
        rows = run_arm(arm, records,
                       lambda m, _model=model: query_openai_compat(local, _model, m),
                       scoring)
        rows_all[arm], summaries[arm] = rows, summarize(rows)

    if args.frontier:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("ERROR: --frontier requires ANTHROPIC_API_KEY")
        import anthropic
        fr = anthropic.Anthropic()
        print(f"\n== frontier ({args.frontier_model}) ==")
        rows = run_arm("frontier", records,
                       lambda m: query_anthropic(fr, args.frontier_model, m),
                       scoring)
        rows_all["frontier"], summaries["frontier"] = rows, summarize(rows)

    out_dir = Path("evals")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "benchmark_demo.json").write_text(json.dumps(
        {"config": {"n": args.n, "base": base, "adapter": adapter,
                    "frontier_model": args.frontier_model if args.frontier else None},
         "summary": summaries, "rows": rows_all}, indent=2))
    write_report(out_dir / "benchmark_demo.md", summaries, len(records),
                 args.frontier_model if args.frontier else None)

    print("\n== summary ==")
    for arm, v in summaries.items():
        print(f"{arm:9s} n={v.get('n', 0):3d} composite={v.get('mean_composite', '—')} "
              f"latency={v.get('median_latency_ms', '—')}ms "
              f"tokens={v.get('median_completion_tokens', '—')}")
    print(f"\nReports: {out_dir}/benchmark_demo.md, {out_dir}/benchmark_demo.json")


if __name__ == "__main__":
    main()
