#!/usr/bin/env python3
"""Prepare MBPP training and MBPP+ evaluation data for code SFT."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from code_verify import clean_completion

DATA_VERSION = "mbpp-function-tests-v1"

SYSTEM_PROMPT = (
    "You are an expert Python programmer. Write a complete, correct Python "
    "solution. Output only executable Python code, with no markdown fences or "
    "explanation."
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_jsonl(records: list[dict], path: Path) -> None:
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _format_prompt(text: str, tests: list[str] | None = None) -> str:
    prompt = text.strip()
    if tests:
        prompt += "\n\nExamples/tests:\n"
        prompt += "\n".join(test.strip() for test in tests if test.strip())
    return prompt


def _training_record(row: dict) -> dict:
    return {
        "conversations": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _format_prompt(row["text"], row.get("test_list") or [])},
            {"role": "assistant", "content": clean_completion(row["code"]).strip()},
        ]
    }


def _mbpp_eval_record(row: dict) -> dict:
    tests = row.get("test_list") or []
    setup = row.get("test_setup_code") or ""
    test_code = (setup.strip() + "\n" if setup.strip() else "") + "\n".join(tests)
    return {
        "task_id": row["task_id"],
        "prompt": _format_prompt(row["text"], tests),
        "test": test_code,
        "canonical_solution": clean_completion(row["code"]).strip(),
        "source": "nlile/mbpp",
    }


def _mbpp_plus_eval_record(row: dict) -> dict:
    imports = "\n".join(row.get("test_imports") or [])
    test_code = (imports + "\n" if imports else "") + row["test"]
    return {
        "task_id": row["task_id"],
        "prompt": _format_prompt(row["prompt"], row.get("test_list") or []),
        "test": test_code,
        "canonical_solution": clean_completion(row["code"]).strip(),
        "source": "evalplus/mbppplus",
    }


def prepare(output_dir: Path) -> dict:
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("datasets not found: pip install datasets")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading nlile/mbpp ...")
    mbpp = load_dataset("nlile/mbpp")
    train_records = [_training_record(row) for row in mbpp["train"]]
    holdout_records = [_training_record(row) for row in mbpp["validation"]]
    mbpp_test_records = [_mbpp_eval_record(row) for row in mbpp["test"]]

    print("Loading evalplus/mbppplus ...")
    mbpp_plus = load_dataset("evalplus/mbppplus", split="test")
    mbpp_plus_records = [_mbpp_plus_eval_record(row) for row in mbpp_plus]

    paths = {
        "train.jsonl": output_dir / "train.jsonl",
        "holdout.jsonl": output_dir / "holdout.jsonl",
        "mbpp_test.jsonl": output_dir / "mbpp_test.jsonl",
        "mbpp_plus_test.jsonl": output_dir / "mbpp_plus_test.jsonl",
    }
    _write_jsonl(train_records, paths["train.jsonl"])
    _write_jsonl(holdout_records, paths["holdout.jsonl"])
    _write_jsonl(mbpp_test_records, paths["mbpp_test.jsonl"])
    _write_jsonl(mbpp_plus_records, paths["mbpp_plus_test.jsonl"])

    manifest = {
        "data_version": DATA_VERSION,
        "train_source": "nlile/mbpp train",
        "holdout_source": "nlile/mbpp validation",
        "eval_source": "evalplus/mbppplus test",
        "train_records": len(train_records),
        "holdout_records": len(holdout_records),
        "mbpp_test_records": len(mbpp_test_records),
        "mbpp_plus_records": len(mbpp_plus_records),
        "files": {
            name: {"sha256": _sha256(path), "lines": sum(1 for _ in path.open())}
            for name, path in paths.items()
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/mbpp"))
    args = parser.parse_args()
    prepare(args.output_dir)


if __name__ == "__main__":
    main()
