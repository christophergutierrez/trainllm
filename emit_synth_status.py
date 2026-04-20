#!/usr/bin/env python3
"""
Emit synth_status.yaml from an eval run JSON.

Produces the mechanically-derivable half of the handoff contract described in
handoff/status_flow.md. The `failure_note` fields (which require semantic LLM
analysis) are left off; everything else is computed from scores, bands,
convention tags, and the training-data manifest.

Usage:
    python emit_synth_status.py                      # latest adapter eval
    python emit_synth_status.py <eval_json_path>     # explicit eval
"""

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import _config

import yaml

cfg = _config.load()

THRESHOLDS = {"excellent": 0.8, "good": 0.6, "partial": 0.4}


def _load_eval(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _adapter_evals(evals_dir: Path, adapter: str) -> list[Path]:
    return sorted(evals_dir.glob(f"*_{adapter}.json"), key=lambda p: p.stat().st_mtime)


def _base_evals(evals_dir: Path, base_model: str) -> list[Path]:
    safe = base_model.replace("/", "_")
    return sorted(evals_dir.glob(f"*_{safe}.json"), key=lambda p: p.stat().st_mtime)


def _priority(avg: float, n: int) -> str:
    if n >= 10 and avg < 0.5:
        return "critical"
    if avg < 0.3:
        return "high"
    if avg < 0.5:
        return "medium"
    return "low"


def _skip_reason(avg: float) -> str:
    if avg >= THRESHOLDS["excellent"]:
        return "EXCELLENT"
    return "GOOD"


def _action(boost: list, avg: float) -> str:
    if boost:
        return "GENERATE_DATA"
    if avg < THRESHOLDS["good"]:
        return "TRAIN"
    if avg >= THRESHOLDS["excellent"]:
        return "DONE"
    return "EVAL"


def _strip_fences(text: str) -> str:
    text = re.sub(r"^```[\w]*\n?", "", text.strip())
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _format_health(results: list) -> dict:
    scored = [r for r in results if r["band"] != "ERROR"]
    func_starts = 0
    boilerplate_cases = []
    for r in scored:
        gen = _strip_fences(r.get("generated", ""))
        if gen.startswith("func "):
            func_starts += 1
        preamble = []
        for line in gen.splitlines():
            if line.strip().startswith("func "):
                break
            preamble.append(line)
        if any(l.strip().startswith(("package ", "import ", "import(")) for l in preamble):
            boilerplate_cases.append({
                "func": r["label"],
                "score": round(r["score"], 3),
                "length_ratio": r.get("length_ratio", 0.0),
            })
    total = max(len(scored), 1)
    return {
        "func_start_pct": round(100 * func_starts / total, 1),
        "boilerplate_detected": len(boilerplate_cases),
        "boilerplate_cases": boilerplate_cases,
    }


def _read_manifest(data_dir: Path) -> dict | None:
    path = data_dir / "manifest.yaml"
    if not path.exists():
        return None
    with open(path) as f:
        return yaml.safe_load(f)


def _uncovered(manifest: dict | None, tested: set[str]) -> list:
    if not manifest:
        return []
    tags: set[str] = set()
    for p in manifest.get("patterns", []) or []:
        tags.update(p.get("tags", []) or [])
    return [{"name": t} for t in sorted(tags - tested)]


def _cycle(manifest: dict | None) -> int | None:
    if not manifest:
        return None
    return (manifest.get("meta") or {}).get("cycle")


def _latest_base_before(base_files: list[Path], current_stem: str) -> Path | None:
    def ts(p: Path) -> str:
        return "_".join(p.stem.split("_")[:2])
    current_ts = "_".join(current_stem.split("_")[:2])
    same_or_after = [p for p in base_files if ts(p) >= current_ts]
    if same_or_after:
        return same_or_after[0]
    return base_files[-1] if base_files else None


def build_status(eval_path: Path) -> dict:
    ev = _load_eval(eval_path)
    meta = ev["meta"]
    summary = ev["summary"]
    results = ev["results"]
    evals_dir = eval_path.parent

    adapter_runs = [p for p in _adapter_evals(evals_dir, meta["model"]) if p != eval_path]
    prev_avg = _load_eval(adapter_runs[-1])["summary"]["avg_score"] if adapter_runs else None

    base_file = _latest_base_before(_base_evals(evals_dir, meta["base_model"]), eval_path.stem)
    base_avg = _load_eval(base_file)["summary"]["avg_score"] if base_file else None

    band_counts = summary["band_counts"]
    boost, skip, tested = [], [], set()
    for c in summary["convention_breakdown"]:
        tested.add(c["convention"])
        entry = {
            "name": c["convention"],
            "avg_score": round(c["avg"], 3),
            "n_holdout_examples": c["n"],
        }
        if c["avg"] < THRESHOLDS["good"]:
            entry["priority"] = _priority(c["avg"], c["n"])
            boost.append(entry)
        else:
            skip.append({
                "name": c["convention"],
                "avg_score": round(c["avg"], 3),
                "reason": _skip_reason(c["avg"]),
            })

    rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    boost.sort(key=lambda b: (rank[b["priority"]], b["avg_score"]))

    manifest = _read_manifest(cfg.data_dir)

    return {
        "meta": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "eval_set": Path(meta["holdout_file"]).stem,
            "cycle": _cycle(manifest),
            "action": _action(boost, summary["avg_score"]),
            "source_eval": eval_path.name,
        },
        "scores": {
            "avg_similarity": round(summary["avg_score"], 3),
            "previous_avg": round(prev_avg, 3) if prev_avg is not None else None,
            "base_model_avg": round(base_avg, 3) if base_avg is not None else None,
            "excellent": band_counts.get("EXCELLENT", 0),
            "good": band_counts.get("GOOD", 0),
            "partial": band_counts.get("PARTIAL", 0),
            "poor": band_counts.get("POOR", 0),
            "total": len(results),
        },
        "boost_conventions": boost,
        "skip_conventions": skip,
        "uncovered_conventions": _uncovered(manifest, tested),
        "format_health": _format_health(results),
    }


def emit(eval_path: Path) -> Path:
    status = build_status(eval_path)
    out_path = eval_path.with_name(eval_path.stem + "_synth_status.yaml")
    with open(out_path, "w") as f:
        yaml.safe_dump(status, f, sort_keys=False, default_flow_style=False, width=100)
    return out_path


def main() -> None:
    if len(sys.argv) > 1:
        eval_path = Path(sys.argv[1]).expanduser()
    else:
        runs = _adapter_evals(cfg.evals_dir, cfg.adapter_name)
        if not runs:
            print(f"No adapter eval JSON found in {cfg.evals_dir}", file=sys.stderr)
            sys.exit(1)
        eval_path = runs[-1]

    out_path = emit(eval_path)
    print(f"Source: {eval_path}")
    print(f"Wrote:  {out_path}")


if __name__ == "__main__":
    main()
