import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lean_context import (
    LeanExample,
    extract_state_tactic,
    format_context_block,
    score_example,
    select_context_examples,
)
from lean_eval import _build_prompt


def test_extracts_sharegpt_state_and_tactic():
    record = {
        "conversations": [
            {
                "from": "human",
                "value": "Given the Lean 4 state:\nn : Nat\n⊢ n + 0 = n\nProvide the next tactical step.",
            },
            {"from": "gpt", "value": "simp"},
        ]
    }

    example = extract_state_tactic(record, index=7)

    assert example.index == 7
    assert example.state_before == "n : Nat\n⊢ n + 0 = n"
    assert example.tactic == "simp"


def test_select_context_examples_is_deterministic_and_relevant():
    query = "n : Nat\n⊢ n + 0 = n"
    examples = [
        LeanExample(index=2, state_before="x : Real\n⊢ x = x", tactic="rfl"),
        LeanExample(index=1, state_before="n : Nat\n⊢ 0 + n = n", tactic="simp"),
        LeanExample(index=0, state_before="n : Nat\n⊢ n + 0 = n", tactic="simp"),
    ]

    selected = select_context_examples(query, examples, n_shots=2)

    assert [ex.index for ex in selected] == [1, 2]
    assert selected[0].score >= selected[1].score


def test_format_context_block_contains_state_and_tactic():
    block = format_context_block([
        LeanExample(index=0, state_before="n : Nat\n⊢ n = n", tactic="rfl", score=0.5)
    ])

    assert "State:\nn : Nat" in block
    assert "Tactic:\nrfl" in block


def test_build_prompt_includes_context_turns_before_query():
    context = [
        LeanExample(index=0, state_before="n : Nat\n⊢ 0 + n = n", tactic="simp", score=0.7)
    ]

    prompt = _build_prompt("n : Nat\n⊢ n + 0 = n", context)

    assert "⊢ 0 + n = n" in prompt
    assert "<|im_start|>assistant\nsimp<|im_end|>" in prompt
    assert prompt.rfind("⊢ n + 0 = n") > prompt.find("⊢ 0 + n = n")


def test_score_example_uses_token_overlap():
    score = score_example(
        "n : Nat\n⊢ n + 0 = n",
        LeanExample(index=0, state_before="n : Nat\n⊢ 0 + n = n", tactic="simp"),
    )

    assert score > 0
