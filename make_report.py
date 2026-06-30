#!/usr/bin/env python3
"""
Generate REPORT.md from evaluation summary JSON files.

Reads reports/lean_eval/*/summary.json (and optionally run_config.json
in each directory) and produces a Markdown report with a benchmark table,
representative examples, and methodology notes.

Usage:
  python make_report.py                               # reads reports/lean_eval/
  python make_report.py --eval-dir path/to/evals     # custom eval dir
  python make_report.py --output reports/REPORT.md   # custom output path
  python make_report.py --dry-run                    # print to stdout
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

# Use CWD as root so the script works both from repo root and from bundle
# (where it lives in scripts/ but the user runs it from the bundle root).
_HERE = Path(__file__).parent
_ROOT = _HERE.parent if _HERE.name == "scripts" else _HERE
# Look in results/ first (GB10 runs), fall back to reports/lean_eval/ (Mac bundle)
_RESULTS = _ROOT / "results"
_LEGACY = _ROOT / "reports" / "lean_eval"
DEFAULT_EVAL_DIR = _RESULTS if _RESULTS.exists() else _LEGACY
DEFAULT_OUTPUT = _ROOT / "reports" / "REPORT.md"

RUN_ORDER = [
    "base-7b", "base-0.5b",
    "target-7b", "target-0.5b", "target-7b-with-context",
    "draft-0.5b", "speculative",
    "frontier-no-context", "frontier-with-context",
    "frontier-api-no-context", "frontier-api-with-context",
]
RUN_LABELS = {
    "base-7b":            "Base 7B (untuned)",
    "base-0.5b":          "Base 0.5B (untuned)",
    "target-7b":          "Fine-tuned 7B",
    "target-0.5b":        "Fine-tuned 0.5B",
    "target-7b-with-context": "Fine-tuned 7B + retrieved context",
    "draft-0.5b":         "Fine-tuned 0.5B (draft only)",
    "speculative":        "Speculative 7B + 0.5B draft",
    "frontier-no-context": "Frontier plan/UI (zero-shot)",
    "frontier-with-context": "Frontier plan/UI (few-shot, training context)",
    "frontier-api-no-context": "Frontier API (zero-shot, explicit opt-in)",
    "frontier-api-with-context": "Frontier API (few-shot, explicit opt-in)",
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
        run_name = summary_path.parent.name
        summary = json.loads(summary_path.read_text())
        cfg_path = summary_path.parent / "run_config.json"
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        pred_path = summary_path.parent / "predictions.jsonl"
        prediction_count = None
        if pred_path.exists():
            with pred_path.open(encoding="utf-8") as f:
                prediction_count = sum(1 for line in f if line.strip())
        runs[run_name] = {"summary": summary, "cfg": cfg}
        runs[run_name]["prediction_count"] = prediction_count
    return runs


def _pct(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v * 100:.1f}%"


def _fmt(v: float | None, digits: int = 1) -> str:
    if v is None:
        return "—"
    return f"{v:.{digits}f}"


def _load_sample_predictions(eval_dir: Path, run_name: str,
                              n_pass: int = 2, n_fail: int = 2) -> tuple[list, list]:
    pred_path = eval_dir / run_name / "predictions.jsonl"
    if not pred_path.exists():
        return [], []
    rows = [json.loads(l) for l in pred_path.read_text().splitlines() if l.strip()]

    def _lean_ok(r: dict) -> bool | None:
        # Support both flat lean_pass (lean_eval.py) and nested lean_result.lean_ok (legacy)
        if "lean_pass" in r:
            return r["lean_pass"]
        lr = r.get("lean_result")
        return lr.get("lean_ok") if lr else None

    passing = [r for r in rows if _lean_ok(r) is True][:n_pass]
    failing = [r for r in rows if _lean_ok(r) is False][:n_fail]
    return passing, failing


def _strip_home(text: str) -> str:
    home = os.path.expanduser("~")
    return re.sub(re.escape(home), "~", text)


def generate_report(
    eval_dir: Path,
    output: Path | None,
    dry_run: bool = False,
) -> str:
    runs = _load_runs(eval_dir)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    commit = _git_commit()

    lines: list[str] = []
    w = lines.append

    w("# Lean Speculative Decoding — Evaluation Report")
    w("")
    w(f"**Date:** {now}  ")
    w(f"**Commit:** {commit}  ")
    w(f"**Machine:** see run_config.json in each run directory  ")
    w("")

    # -----------------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------------
    w("## Dataset")
    w("")
    w("Source: `liminho123/lean4-stat-learning-theory-novel`")
    w("")
    w("| Split | Records |")
    w("|-------|---------|")
    data_dir = _ROOT / "data" / "lean_stat"
    counts: dict[str, str] = {}
    for split in ("train", "valid", "test"):
        f = data_dir / f"{split}.jsonl"
        counts[split] = (str(sum(1 for l in f.read_text().splitlines() if l.strip()))
                         if f.exists() else "—")
    w(f"| Train | {counts['train']} |")
    w(f"| Valid | {counts['valid']} |")
    w(f"| Test  | {counts['test']} |")
    w("")
    w("All runs use the same deterministic test split (seed=42).")
    w("")

    # -----------------------------------------------------------------------
    # Training settings
    # -----------------------------------------------------------------------
    w("## Training Settings")
    w("")
    w("| Setting | 0.5B draft | 7B target |")
    w("|---------|------------|-----------|")
    w("| Base model | Qwen2.5-Coder-0.5B-Instruct | Qwen2.5-Coder-7B-Instruct |")
    w("| LoRA rank | 32 | 32 |")
    w("| LoRA alpha | 64 | 64 |")
    w("| rsLoRA | yes | yes |")
    w("| Max steps | 2000 | 2000 |")
    w("| Learning rate | 2e-4 | 1e-4 |")
    w("| Batch size (effective) | 16 | 16 |")
    w("| FP8 loading | disabled | disabled |")
    w("| Loss mask | responses only | responses only |")
    w("")

    # -----------------------------------------------------------------------
    # Benchmark table
    # -----------------------------------------------------------------------
    w("## Benchmark Results")
    w("")
    if not runs:
        w("> _No evaluation runs found in_ `{}`_. "
          "Run `lean_eval.py` to generate results._".format(
              str(eval_dir).replace(str(_HERE) + "/", "")))
    else:
        w("| Run | Records | Evaluated | Compile Pass | Safety Fail | Latency (s) | Tokens/s | Mean In Toks | Mean Out Toks | Notes |")
        w("|-----|---------|-----------|-------------|-------------|------------|----------|-------------|--------------|-------|")
        for key in RUN_ORDER + [k for k in sorted(runs) if k not in RUN_ORDER]:
            if key not in runs:
                continue
            s = runs[key]["summary"]
            label = RUN_LABELS.get(key, key)
            n = s.get("n_total", "—")
            n_eval = s.get("n_evaluated", n)  # n_evaluated added in lean_eval.py v2
            pass_rate = _pct(s.get("compile_pass_rate"))
            safety_fail = s.get("n_safety_fail", "—")
            latency = _fmt(s.get("mean_elapsed_s"))
            tps = _fmt(s.get("mean_tps"))
            # API runs record per-call token counts; local runs do not
            in_toks = _fmt(s.get("mean_input_tokens"), 0)
            out_toks = _fmt(s.get("mean_output_tokens"), 0)
            notes = ""
            if s.get("draft_model"):
                nd = s.get("num_draft_tokens", "?")
                notes = f"{nd} draft tokens"
            elif s.get("mode") == "with-context":
                notes = f"{s.get('n_shots', '?')}-shot"
            pred_count = runs[key].get("prediction_count")
            if pred_count is not None and isinstance(n_eval, int) and pred_count != n_eval:
                notes = (notes + "; " if notes else "") + f"INCONSISTENT: {pred_count} predictions"
            w(f"| {label} | {n} | {n_eval} | {pass_rate} | {safety_fail} | {latency} | {tps} | {in_toks} | {out_toks} | {notes} |")
    w("")

    # -----------------------------------------------------------------------
    # Representative examples
    # -----------------------------------------------------------------------
    primary_run = next(
        (k for k in ["speculative", "target-7b"] if k in runs),
        next(iter(runs), None),
    )
    if primary_run:
        passing, failing = _load_sample_predictions(eval_dir, primary_run)
        if passing:
            w("## Representative Successes")
            w("")
            w(f"_From run: {primary_run}_")
            w("")
            for r in passing:
                w(f"**State:**")
                w("```")
                w(r.get("state_before", ""))
                w("```")
                w(f"**Generated:** `{r.get('generated_tactic', '')}` "
                  f"_(expected: `{r.get('expected_tactic', '')}`)_")
                w("")
        if failing:
            w("## Representative Failures")
            w("")
            w(f"_From run: {primary_run}_")
            w("")
            for r in failing:
                w(f"**State:**")
                w("```")
                w(r.get("state_before", ""))
                w("```")
                w(f"**Generated:** `{r.get('generated_tactic', '')}`  ")
                lean_stderr = r.get("lean_stderr") or (r.get("lean_result") or {}).get("stderr", "")
                if lean_stderr:
                    w(f"**Lean error:** `{lean_stderr[:120].strip()}`")
                w("")

    # -----------------------------------------------------------------------
    # Speed note
    # -----------------------------------------------------------------------
    w("## Speed Notes")
    w("")
    w("Tokens/sec figures are wall-clock end-to-end including tokenization, "
      "not peak throughput. They are machine-specific and should not be "
      "generalized across hardware.")
    w("")
    spec_run = runs.get("speculative", {}).get("summary", {})
    tgt_run = runs.get("target-7b", {}).get("summary", {})
    if spec_run.get("mean_tps") and tgt_run.get("mean_tps"):
        ratio = spec_run["mean_tps"] / tgt_run["mean_tps"]
        w(f"Observed speculative speedup vs target-only: "
          f"**{ratio:.2f}×** ({spec_run['mean_tps']:.1f} vs "
          f"{tgt_run['mean_tps']:.1f} tokens/sec). "
          f"This reflects draft acceptance rate on this machine and test set; "
          f"it does not imply correctness improvement.")
    else:
        w("_Speedup figure not available — run both target-only and speculative "
          "evaluations and re-run `make_report.py`._")
    w("")

    # -----------------------------------------------------------------------
    # Limitations
    # -----------------------------------------------------------------------
    w("## Limitations")
    w("")
    w("- **Verification coverage:** Lean compilation is attempted only for "
      "goal states that can be reconstructed as standalone `example` blocks "
      "from the tactic state string alone. States with universe metavariables, "
      "instance dummies (`inst✝`), or complex dependencies may be skipped "
      "(counted in `n_lean_skip`).")
    w("- **Correctness metric:** Compile pass rate measures whether the "
      "generated tactic type-checks in a reconstructed context, not whether it "
      "is the best proof step. A tactic that type-checks in isolation may not "
      "generalize.")
    w("- **Dataset scope:** The dataset covers `lean4-stat-learning-theory-novel`; "
      "results may not transfer to other Lean libraries or proof styles.")
    w("- **Speculative correctness:** Speculative decoding is mathematically "
      "equivalent to target-only sampling when the draft is accepted. Any "
      "pass-rate difference between the two runs reflects sampling variance, "
      "not a quality difference.")
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
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR,
                   help="Directory containing run subdirectories with summary.json")
    p.add_argument("--output", type=Path, default=None,
                   help=f"Output path (default: {DEFAULT_OUTPUT})")
    p.add_argument("--dry-run", action="store_true",
                   help="Print to stdout instead of writing a file")
    args = p.parse_args()
    generate_report(args.eval_dir, args.output, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
