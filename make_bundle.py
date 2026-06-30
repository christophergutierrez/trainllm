#!/usr/bin/env python3
"""
Assemble the lean-speculative-bundle handoff package.

Copies scripts, docs, sample data, eval harness, and completed reports
into a self-contained directory that a Mac user can run without training
dependencies or absolute paths.

Fused model directories (fused-7b-lean/, fused-0.5b-lean/) are NOT copied
automatically because they are large. They must be created by running
`mlx_lm.fuse` on Mac and then placed or symlinked into the bundle.

Bundle layout:
  lean-speculative-bundle/
    README.md             <- from docs/lean_speculative/README.md
    RUNBOOK.md            <- from docs/lean_speculative/RUNBOOK.md
    FULL_EVAL.md          <- from docs/lean_speculative/FULL_EVAL.md
    scripts/
      lean_verify.py
      lean_eval.py
      make_report.py
    data/
      sample/
        fixture_5.jsonl
        test_20.jsonl     (20-record sample from test split)
    eval/
      lean_harness/
        Fixture.lean
        fixture_5.jsonl
    reports/              (populated after eval runs; empty placeholder)
    fused-7b-lean/        <- NOT created here; symlinked or copied manually
    fused-0.5b-lean/      <- NOT created here; symlinked or copied manually

Usage:
  python make_bundle.py                          # assemble bundle
  python make_bundle.py --validate-only          # check constraints only
  python make_bundle.py --bundle-dir /tmp/pkg    # custom output dir
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).parent
DEFAULT_BUNDLE = _HERE / "lean-speculative-bundle"

# Files / dirs that must not appear in the bundle (detected recursively).
FORBIDDEN_PATTERNS = [
    re.compile(r"/__pycache__/"),
    re.compile(r"/checkpoint-\d+"),
    re.compile(r"/\.cache/"),
    re.compile(r"/models/hf/"),
    # safetensors are allowed in fused-*/ (model weights) but not in adapters/ or lora/
    re.compile(r"/(lora|adapters)/.*\.safetensors$"),
]

# Text files to scan for absolute /home/... paths.
TEXT_EXTS = {".py", ".md", ".json", ".yaml", ".yml", ".txt", ".lean"}

ABS_PATH_RE = re.compile(r"/home/[a-z][a-zA-Z0-9_-]+/")


def _check_no_abs_paths(path: Path) -> list[str]:
    issues = []
    for f in path.rglob("*"):
        if f.is_file() and f.suffix in TEXT_EXTS:
            try:
                content = f.read_text(errors="replace")
            except Exception:
                continue
            for i, line in enumerate(content.splitlines(), 1):
                if ABS_PATH_RE.search(line):
                    issues.append(f"{f.relative_to(path)}:{i}: {line.strip()[:80]}")
    return issues


def _check_no_forbidden(path: Path) -> list[str]:
    issues = []
    for f in path.rglob("*"):
        rel = "/" + str(f.relative_to(path)) + ("/" if f.is_dir() else "")
        for pat in FORBIDDEN_PATTERNS:
            if pat.search(rel):
                issues.append(str(f.relative_to(path)))
    return issues


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  copied  {dst.relative_to(dst.parent.parent.parent)}")


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"  copied  {dst.name}/ ({sum(1 for _ in dst.rglob('*'))} files)")


def _sample_test_data(src: Path, out: Path, n: int = 20) -> None:
    """Copy first n records from test.jsonl as a sample."""
    out.parent.mkdir(parents=True, exist_ok=True)
    records = []
    if src.exists():
        with open(src) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(line)
                    if len(records) >= n:
                        break
    out.write_text("\n".join(records) + "\n" if records else "")
    print(f"  sampled {len(records)} records -> {out.name}")


def assemble(bundle_dir: Path, validate_only: bool = False) -> bool:
    if not validate_only:
        bundle_dir.mkdir(parents=True, exist_ok=True)

    ok = True
    errors: list[str] = []

    # -----------------------------------------------------------------------
    # Docs
    # -----------------------------------------------------------------------
    docs_src = _HERE / "docs" / "lean_speculative"
    doc_map = {
        "BUNDLE_README.md": "README.md",  # bundle gets its own standalone README
        "RUNBOOK.md": "RUNBOOK.md",
        "FULL_EVAL.md": "FULL_EVAL.md",
    }
    for src_name, dst_name in doc_map.items():
        src = docs_src / src_name
        if not src.exists():
            errors.append(f"missing doc: {src.relative_to(_HERE)}")
            ok = False
            continue
        if not validate_only:
            _copy(src, bundle_dir / dst_name)

    # -----------------------------------------------------------------------
    # Scripts (also include make_bundle.py so the bundle is self-documenting)
    # -----------------------------------------------------------------------
    scripts_dst = bundle_dir / "scripts"
    for script in ["lean_verify.py", "lean_eval.py", "make_report.py", "make_bundle.py"]:
        src = _HERE / script
        if not src.exists():
            errors.append(f"missing script: {script}")
            ok = False
            continue
        if not validate_only:
            _copy(src, scripts_dst / script)

    # -----------------------------------------------------------------------
    # Eval harness
    # -----------------------------------------------------------------------
    harness_src = _HERE / "eval" / "lean_harness"
    if harness_src.exists() and not validate_only:
        _copy_tree(harness_src, bundle_dir / "eval" / "lean_harness")

    # -----------------------------------------------------------------------
    # Data — full test split + samples
    # -----------------------------------------------------------------------
    if not validate_only:
        fixture_src = harness_src / "fixture_5.jsonl"
        if fixture_src.exists():
            _copy(fixture_src, bundle_dir / "data" / "sample" / "fixture_5.jsonl")

        test_src = _HERE / "data" / "lean_stat" / "test.jsonl"
        # Include the full test set so lean_eval.py default path works
        if test_src.exists():
            _copy(test_src, bundle_dir / "data" / "lean_stat" / "test.jsonl")
        else:
            errors.append("missing data/lean_stat/test.jsonl — run prepare_lean_data.py first")
            ok = False
        _sample_test_data(test_src, bundle_dir / "data" / "sample" / "test_20.jsonl")

    # -----------------------------------------------------------------------
    # Reports placeholder
    # -----------------------------------------------------------------------
    if not validate_only:
        reports_dir = bundle_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        (reports_dir / ".gitkeep").touch()

        # Copy any completed reports
        src_reports = _HERE / "reports"
        if src_reports.exists():
            for p in src_reports.rglob("*.md"):
                _copy(p, reports_dir / p.relative_to(src_reports))
            for p in src_reports.rglob("summary.json"):
                _copy(p, reports_dir / p.relative_to(src_reports))

    # -----------------------------------------------------------------------
    # Fused model stubs (placeholders, not the actual weights)
    # -----------------------------------------------------------------------
    if not validate_only:
        for model_name in ["fused-7b-lean", "fused-0.5b-lean"]:
            stub_dir = bundle_dir / model_name
            model_src = _HERE / model_name
            if model_src.exists():
                _copy_tree(model_src, stub_dir)
            else:
                stub_dir.mkdir(parents=True, exist_ok=True)
                (stub_dir / "PLACEHOLDER.md").write_text(
                    f"# {model_name}\n\n"
                    "This directory will contain the fused MLX model.\n\n"
                    "Run on Mac after copying the MLX adapters:\n\n"
                    f"```bash\n"
                    f"python3 -m mlx_lm.fuse \\\n"
                    f"  --model {'Qwen/Qwen2.5-Coder-7B-Instruct' if '7b' in model_name else 'Qwen/Qwen2.5-Coder-0.5B-Instruct'} \\\n"
                    f"  --adapter-path adapters/{model_name.replace('fused-', '').replace('-lean', '-lean-mlx')} \\\n"
                    f"  --save-path {model_name}\n"
                    f"```\n"
                )
                print(f"  created {model_name}/ (placeholder — fuse on Mac)")

    # -----------------------------------------------------------------------
    # Validate the assembled bundle
    # -----------------------------------------------------------------------
    if not validate_only and bundle_dir.exists():
        print("\nValidating bundle...")
        forbidden = _check_no_forbidden(bundle_dir)
        for issue in forbidden:
            errors.append(f"forbidden pattern in bundle: {issue}")
        abs_issues = _check_no_abs_paths(bundle_dir)
        for issue in abs_issues:
            errors.append(f"absolute path in bundle: {issue}")

    # Validate source files too
    for script in ["lean_verify.py", "lean_eval.py", "make_report.py"]:
        src = _HERE / script
        if src.exists():
            content = src.read_text()
            if ABS_PATH_RE.search(content):
                errors.append(f"absolute path in source: {script}")

    if errors:
        ok = False
        print("\nErrors:")
        for e in errors:
            print(f"  ERROR: {e}")
    else:
        print("\nAll checks passed.")

    return ok


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE,
                   help=f"Output directory (default: {DEFAULT_BUNDLE.name})")
    p.add_argument("--validate-only", action="store_true",
                   help="Check source files only; do not write the bundle")
    args = p.parse_args()

    print(f"{'Validating' if args.validate_only else 'Assembling'} bundle "
          f"-> {args.bundle_dir}")
    ok = assemble(args.bundle_dir, validate_only=args.validate_only)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
