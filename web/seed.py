#!/usr/bin/env python3
"""Fixture generator/importer for web dashboard development.

Usage:
    python web/seed.py              # Generate synthetic fixtures
    python web/seed.py --from X.tar.gz  # Import from a tarball
    python web/seed.py --export     # Export current pipeline state as tarball
"""

import argparse
import json
import random
import shutil
import sys
import tarfile
from datetime import datetime, timedelta
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"

FIXTURE_FILES = [
    "convergence.json",
    "eval_sample.json",
    "eval_sample_llmjudge.json",
    "events.jsonl",
    "config.yaml",
]


def generate_synthetic():
    """Generate plausible synthetic fixtures for development."""
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    _gen_convergence()
    _gen_eval()
    _gen_llmjudge()
    _gen_events()
    _gen_config()

    print(f"Generated synthetic fixtures in {FIXTURES_DIR}/")
    for f in FIXTURE_FILES:
        p = FIXTURES_DIR / f
        if p.exists():
            print(f"  {f} ({p.stat().st_size} bytes)")


def _gen_convergence():
    steps = 400
    losses = []
    loss = 1.4
    for i in range(0, steps, 10):
        loss = max(0.02, loss - random.uniform(0.001, 0.01))
        losses.append((i, round(loss, 5), round(i / (steps / 3), 2)))

    data = {
        "first_loss": losses[0][1],
        "final_loss": losses[-1][1],
        "best_loss": min(l[1] for l in losses),
        "best_step": max(range(len(losses)), key=lambda i: -losses[i][1]),
        "total_steps": steps,
        "final_epoch": losses[-1][2],
        "convergence_rate": round((losses[0][1] - losses[-1][1]) / steps, 7),
        "stopped_early": False,
        "loss_history": losses,
    }
    (FIXTURES_DIR / "convergence.json").write_text(json.dumps(data, indent=2) + "\n")


def _gen_eval():
    bands = ["EXCELLENT", "GOOD", "PARTIAL", "POOR"]
    conventions = ["campaigns", "pixels", "networks", "measurements", "audiences"]
    records = []
    for i in range(30):
        sim = round(random.uniform(0.2, 0.95), 4)
        struct = round(random.uniform(0.1, 1.0), 4)
        comp = round(sim * 0.3 + struct * 0.7, 4)
        b = "EXCELLENT" if comp >= 0.8 else "GOOD" if comp >= 0.6 else "PARTIAL" if comp >= 0.4 else "POOR"
        records.append({
            "id": f"holdout_{i+1:03d}",
            "question": f"Get {random.choice(conventions)} with page size {random.randint(5,50)}",
            "expected": f'```json\n{{"endpoint": "GET /{random.choice(conventions)}", "params": {{"pageSize": {random.randint(5,50)}}}}}\n```',
            "generated": f'```json\n{{"endpoint": "GET /{random.choice(conventions)}", "params": {{"pageSize": {random.randint(5,50)}}}}}\n```',
            "score": sim,
            "structural_score": struct,
            "composite_score": comp,
            "band": b,
            "conventions": [random.choice(conventions)],
        })

    band_counts = {}
    for r in records:
        band_counts[r["band"]] = band_counts.get(r["band"], 0) + 1

    scores = [r["composite_score"] for r in records]
    data = {
        "meta": {
            "timestamp": datetime.now().strftime("%Y-%m-%d_%H%M"),
            "model": "test-adapter",
        },
        "summary": {
            "avg_score": round(sum(r["score"] for r in records) / len(records), 4),
            "avg_structural_score": round(sum(r["structural_score"] for r in records) / len(records), 4),
            "avg_composite_score": round(sum(scores) / len(scores), 4),
            "band_counts": band_counts,
        },
        "results": records,
    }
    (FIXTURES_DIR / "eval_sample.json").write_text(json.dumps(data, indent=2) + "\n")


def _gen_llmjudge():
    records = []
    for i in range(10):
        score = round(random.uniform(0.2, 0.9), 2)
        records.append({
            "id": f"holdout_{i+1:03d}",
            "judge_score": score,
            "band": "EXCELLENT" if score >= 0.8 else "GOOD" if score >= 0.6 else "PARTIAL" if score >= 0.4 else "POOR",
            "explanation": f"Synthetic judge explanation for record {i+1}.",
        })
    data = {
        "meta": {"model": "claude-haiku-4-5-20251001"},
        "summary": {
            "avg_score": round(sum(r["judge_score"] for r in records) / len(records), 4),
        },
        "results": records,
    }
    (FIXTURES_DIR / "eval_sample_llmjudge.json").write_text(json.dumps(data, indent=2) + "\n")


def _gen_events():
    lines = []
    base_time = datetime.now() - timedelta(minutes=30)
    steps_info = [
        ("prepare", 5),
        ("train", 300),
        ("serve", 15),
        ("eval", 60),
        ("judge", 20),
        ("report", 2),
    ]
    offset = 0
    for step_name, duration in steps_info:
        start_ts = (base_time + timedelta(seconds=offset)).isoformat()
        lines.append(json.dumps({"event": "step_start", "step": step_name, "timestamp": start_ts}))
        if step_name == "train":
            for s in range(0, 400, 10):
                loss = max(0.02, 1.4 - s * 0.003 + random.uniform(-0.01, 0.01))
                lr = 2e-4 if s < 280 else 2e-4 * (1 - 0.9 * (s - 280) / 120)
                lines.append(json.dumps({
                    "event": "loss",
                    "step": s,
                    "value": round(loss, 5),
                    "lr": round(lr, 8),
                    "timestamp": (base_time + timedelta(seconds=offset + s * 0.75)).isoformat(),
                }))
        offset += duration
        end_ts = (base_time + timedelta(seconds=offset)).isoformat()
        lines.append(json.dumps({"event": "step_end", "step": step_name, "duration_sec": duration, "timestamp": end_ts}))

    (FIXTURES_DIR / "events.jsonl").write_text("\n".join(lines) + "\n")


def _gen_config():
    config = {
        "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "adapter_name": "test-adapter",
        "chat_template": "qwen-2.5",
        "training": {
            "max_seq_length": 512,
            "lora_rank": 16,
            "lora_alpha": 32,
            "batch_size": 8,
            "gradient_accumulation_steps": 4,
            "warmup_steps": 20,
            "max_steps": 400,
            "learning_rate": 2e-4,
            "lr_scheduler": "wsd",
            "neftune_noise_alpha": 5,
            "train_on_responses_only": True,
        },
    }
    import yaml
    (FIXTURES_DIR / "config.yaml").write_text(yaml.dump(config, default_flow_style=False))


def import_from(tarball_path: str):
    """Import fixtures from a tarball."""
    path = Path(tarball_path)
    if not path.exists():
        sys.exit(f"File not found: {path}")

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    with tarfile.open(path, "r:gz") as tar:
        tar.extractall(FIXTURES_DIR, filter="data")

    print(f"Imported fixtures from {path} → {FIXTURES_DIR}/")


def export_fixtures():
    """Export current pipeline state as a shareable fixture tarball."""
    base_dir = Path(__file__).parent.parent
    sources = {
        "config.yaml": base_dir / "config.yaml",
    }

    # Find most recent eval
    evals_dir = base_dir / "evals"
    if evals_dir.exists():
        eval_files = sorted(evals_dir.glob("*.json"), reverse=True)
        for ef in eval_files:
            if "_llmjudge" not in ef.name and "_synth" not in ef.name:
                sources["eval_sample.json"] = ef
                judge = evals_dir / f"{ef.stem}_llmjudge.json"
                if judge.exists():
                    sources["eval_sample_llmjudge.json"] = judge
                break

    # Find convergence
    lora_dir = base_dir / "lora"
    if lora_dir.exists():
        for conv in lora_dir.rglob("convergence.json"):
            sources["convergence.json"] = conv
            break

    # Events
    events_file = Path("/tmp/trainllm_events.jsonl")
    if events_file.exists():
        sources["events.jsonl"] = events_file

    ts = datetime.now().strftime("%Y-%m-%d")
    out_path = Path(f"fixtures-{ts}.tar.gz")

    with tarfile.open(out_path, "w:gz") as tar:
        for name, src in sources.items():
            if src.exists():
                tar.add(src, arcname=name)

    print(f"Exported fixtures → {out_path} ({out_path.stat().st_size} bytes)")
    print(f"Files included:")
    for name, src in sources.items():
        if src.exists():
            print(f"  {name} ← {src}")


def main():
    parser = argparse.ArgumentParser(description="Manage web dashboard fixtures")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--from", dest="from_path", help="Import fixtures from a tarball")
    group.add_argument("--export", action="store_true", help="Export current pipeline state")
    args = parser.parse_args()

    if args.from_path:
        import_from(args.from_path)
    elif args.export:
        export_fixtures()
    else:
        generate_synthetic()


if __name__ == "__main__":
    main()
