#!/usr/bin/env python3
"""
Generate REPORT.md from APPS evaluation summary JSON files.

Usage:
    python3 apps_report.py                          # reads results/apps/
    python3 apps_report.py --eval-dir path/to/evals
    python3 apps_report.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).parent
_ROOT = _HERE.parent if _HERE.name == "scripts" else _HERE
DEFAULT_EVAL_DIR = _ROOT / "results" / "apps"
DEFAULT_OUTPUT   = _ROOT / "reports" / "apps" / "REPORT.md"

TIERS = ("introductory", "interview", "competition")

RUN_ORDER = [
    "base-7b", "base-0.5b",
    "target-7b", "target-0.5b",
    "target-7b-context",
    "draft-0.5b", "speculative",
    "frontier-no-context", "frontier-with-context",
]
RUN_LABELS = {
    "base-7b":             "Base 7B (untuned)",
    "base-0.5b":           "Base 0.5B (untuned)",
    "target-7b":           "Fine-tuned 7B",
    "target-0.5b":         "Fine-tuned 0.5B",
    "target-7b-context":   "Fine-tuned 7B + retrieved context",
    "draft-0.5b":          "Fine-tuned 0.5B (draft only)",
    "speculative":         "Speculative 7B + 0.5B draft",
    "frontier-no-context": "Frontier API (zero-shot)",
    "frontier-with-context": "Frontier API (few-shot)",
}


def _git_commit() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_HERE, stderr=subprocess.DEVNULL, text=True,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=_HERE, stderr=subprocess.DEVNULL, text=True,
        ).strip()
        return sha + (" (dirty)" if dirty else "")
    except Exception:
        return "unknown"


def _load_runs(eval_dir: Path) -> dict[str, dict]:
    runs: dict[str, dict] = {}
    if not eval_dir.exists():
        return runs
    for summary_path in sorted(eval_dir.glob("*/summary.json")):
        name = summary_path.parent.name
        summary = json.loads(summary_path.read_text())
        cfg_path = summary_path.parent / "run_config.json"
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}

        # Per-tier breakdown from predictions.jsonl
        tier_counts: dict[str, dict] = {t: {"pass": 0, "total": 0} for t in TIERS}
        pred_path = summary_path.parent / "predictions.jsonl"
        if pred_path.exists():
            for line in pred_path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                tier = row.get("difficulty", "")
                if tier in tier_counts:
                    tier_counts[tier]["total"] += 1
                    if row.get("passed"):
                        tier_counts[tier]["pass"] += 1

        runs[name] = {"summary": summary, "cfg": cfg, "tier_counts": tier_counts}
    return runs


def _pct(v: float | None) -> str:
    return "—" if v is None else f"{v * 100:.1f}%"


def _fmt(v: float | None, d: int = 1) -> str:
    return "—" if v is None else f"{v:.{d}f}"


def _strip_home(text: str) -> str:
    home = os.path.expanduser("~")
    return re.sub(re.escape(home), "~", text)


def generate_report(eval_dir: Path, output: Path | None, dry_run: bool = False) -> str:
    runs = _load_runs(eval_dir)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    commit = _git_commit()

    lines: list[str] = []
    w = lines.append

    w("# APPS Speculative Decoding — Evaluation Report")
    w("")
    w(f"**Date:** {now}  ")
    w(f"**Commit:** {commit}  ")
    w("")

    # Dataset
    w("## Dataset")
    w("")
    w("Source: `codeparrot/apps`  ")
    w("Split: test holdout — 200 problems per tier, seed=42  ")
    w("")
    w("| Tier | Problems |")
    w("|------|----------|")
    for tier in TIERS:
        path = _ROOT / "data" / "apps" / f"eval_{tier}.jsonl"
        n = sum(1 for l in path.read_text().splitlines() if l.strip()) if path.exists() else "—"
        w(f"| {tier.capitalize()} | {n} |")
    w("")

    # Main benchmark table
    w("## Benchmark Results")
    w("")
    if not runs:
        w("> _No results found in_ `{}`".format(eval_dir))
    else:
        w("| Run | Total | Pass% | Intro Pass% | Interview Pass% | Competition Pass% | Latency (s) | Tokens/s | Notes |")
        w("|-----|-------|-------|-------------|-----------------|-------------------|------------|----------|-------|")
        for key in RUN_ORDER + [k for k in sorted(runs) if k not in RUN_ORDER]:
            if key not in runs:
                continue
            s = runs[key]["summary"]
            tc = runs[key]["tier_counts"]
            label = RUN_LABELS.get(key, key)
            n_total = s.get("n_total", "—")
            pass_rate = _pct(s.get("pass_rate"))
            latency = _fmt(s.get("mean_elapsed_s"))
            tps = _fmt(s.get("mean_tps"))
            notes = ""
            if s.get("draft_model"):
                notes = f"{s.get('num_draft_tokens','?')} draft tokens"
            elif s.get("mode") == "with-context":
                notes = f"{s.get('n_shots','?')}-shot"
            if s.get("n_scan_blocked", 0) > 0:
                notes = (notes + "; " if notes else "") + f"{s['n_scan_blocked']} scan-blocked"

            def tier_pct(tier: str) -> str:
                c = tc.get(tier, {})
                t, pa = c.get("total", 0), c.get("pass", 0)
                return f"{pa/t*100:.0f}% ({pa}/{t})" if t else "—"

            w(f"| {label} | {n_total} | {pass_rate} | {tier_pct('introductory')} | "
              f"{tier_pct('interview')} | {tier_pct('competition')} | "
              f"{latency} | {tps} | {notes} |")
    w("")

    # Speed note
    w("## Speed Notes")
    w("")
    w("Tokens/sec is wall-clock end-to-end including tokenisation. Machine-specific.")
    w("")
    spec = runs.get("speculative", {}).get("summary", {})
    tgt  = runs.get("target-7b",   {}).get("summary", {})
    if spec.get("mean_tps") and tgt.get("mean_tps"):
        ratio = spec["mean_tps"] / tgt["mean_tps"]
        w(f"Speculative vs target-only speedup: **{ratio:.2f}×** "
          f"({spec['mean_tps']:.1f} vs {tgt['mean_tps']:.1f} tok/s). "
          f"Pass rates should be equal; any difference is sampling variance.")
    else:
        w("_Speedup not available — run both `target-7b` and `speculative` then re-run this script._")
    w("")

    # Limitations
    w("## Limitations")
    w("")
    w("- **Correctness metric:** Pass@1 with temperature=0. Reports whether the first generated "
      "solution passes all sampled test cases; does not measure pass@k or test-case coverage.")
    w("- **Test-case cap:** Up to 5 test cases per problem are run. Problems with only trivial "
      "test cases may show inflated pass rates.")
    w("- **Code scan:** Solutions blocked by the safety scan count as skipped, not failed. "
      "A high scan-blocked rate indicates the model generates unsafe patterns.")
    w("- **Speculative correctness:** Draft-plus-target is mathematically equivalent to "
      "target-only sampling under the acceptance criterion. Pass-rate differences "
      "are sampling noise, not quality differences.")
    w("")

    report = "\n".join(lines)
    report = _strip_home(report)

    if dry_run:
        print(report)
    else:
        if output is None:
            output = DEFAULT_OUTPUT
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report)
        print(f"Report written to {output}")

    return report


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    generate_report(args.eval_dir, args.output, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
