import json
from pathlib import Path

from lean_eval import run_eval


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def test_run_eval_counts_only_generated_records_for_timing(tmp_path):
    test_path = tmp_path / "test.jsonl"
    _write_jsonl(
        test_path,
        [
            {
                "conversations": [
                    {
                        "from": "human",
                        "value": "Given the Lean 4 state:\n⊢ True\nProvide the next tactical step.",
                    },
                    {"from": "gpt", "value": "trivial"},
                ]
            },
            {
                "conversations": [
                    {
                        "from": "human",
                        "value": (
                            "Given the Lean 4 state:\n"
                            "h : " + ("Nat -> " * 100) + "Nat\n"
                            "⊢ True\n"
                            "Provide the next tactical step."
                        ),
                    },
                    {"from": "gpt", "value": "trivial"},
                ]
            },
        ],
    )

    out_dir = tmp_path / "out"
    summary = run_eval(
        model_path="mock",
        test_path=test_path,
        out_dir=out_dir,
        backend="mock",
        skip_lean=True,
        max_state_chars=50,
    )

    predictions = [
        json.loads(line)
        for line in (out_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert summary["n_total"] == 2
    assert summary["n_skipped_size"] == 1
    assert summary["n_evaluated"] == 1
    assert len(predictions) == 1
    assert summary["mean_elapsed_s"] > 0
