"""Tests verifying that prepare_lean_data.py output records pass through the
Qwen/ChatML training pipeline used by train.py.

All tests use in-memory fixtures and do NOT require model weights or internet
access.  The chat template is applied via jinja2 using the same ChatML markup
that Qwen models use.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from prepare_lean_data import make_record

# ── Minimal ChatML template (mirrors Qwen's chat template) ──────────────────

CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n"
    "{{ message['content'] }}"
    "<|im_end|>\n"
    "{% endfor %}"
)

INSTRUCTION_PART = "<|im_start|>user\n"
RESPONSE_PART = "<|im_start|>assistant\n"


def _render(messages: list[dict]) -> str:
    """Apply the ChatML jinja2 template to a list of {role, content} dicts."""
    from jinja2 import Environment

    env = Environment()
    tmpl = env.from_string(CHATML_TEMPLATE)
    return tmpl.render(messages=messages)


def _standardize(record: dict) -> list[dict]:
    """Convert a single make_record() record to [{role, content}] messages.

    Replicates what standardize_sharegpt() does for the role-alias mapping
    without requiring a HuggingFace Dataset.  This keeps tests fast and free
    of multiprocessing overhead while testing the same logical transformation.
    """
    _alias = {"human": "user", "gpt": "assistant"}
    return [
        {"role": _alias[msg["from"]], "content": msg["value"]}
        for msg in record["conversations"]
    ]


def _standardize_via_unsloth(records: list[dict]):
    """Run standardize_sharegpt from unsloth on a batch of records.

    Requires multiple distinct records so unsloth's heuristic (fewer unique
    values → role key) correctly identifies 'from' as the role column.
    Returns the first result.
    """
    from datasets import Dataset
    from unsloth.chat_templates import standardize_sharegpt

    dataset = Dataset.from_list(records)
    result = standardize_sharegpt(dataset)
    return result[0]


# ── Test 1 ───────────────────────────────────────────────────────────────────

class TestStandardizeSharegptConvertsFormat:
    """standardize_sharegpt maps conversations/from/value → role/content."""

    def test_standardize_sharegpt_converts_format(self):
        # Need several distinct records so unsloth's field-detection heuristic
        # (fewer unique values = role column) picks 'from' over 'value'.
        records = [make_record(f"⊢ n = {i}", f"tactic_{i}") for i in range(6)]
        first = _standardize_via_unsloth(records)

        messages = first["conversations"]
        assert len(messages) == 2, "Expected exactly 2 messages"

        user_msg = messages[0]
        asst_msg = messages[1]

        # Keys must be role/content, not from/value
        assert "role" in user_msg and "content" in user_msg
        assert "from" not in user_msg and "value" not in user_msg

        # Role values must be standardised
        assert user_msg["role"] == "user"
        assert asst_msg["role"] == "assistant"

    def test_user_content_preserved(self):
        records = [make_record(f"⊢ n = {i}", f"tac_{i}") for i in range(6)]
        first = _standardize_via_unsloth(records)
        user_content = first["conversations"][0]["content"]
        assert "⊢ n = 0" in user_content
        assert "Lean 4 state" in user_content

    def test_assistant_content_preserved(self):
        records = [make_record(f"⊢ n = {i}", f"tac_{i}") for i in range(6)]
        first = _standardize_via_unsloth(records)
        asst_content = first["conversations"][1]["content"]
        assert asst_content == "tac_0"


# ── Test 2 ───────────────────────────────────────────────────────────────────

class TestRenderedPromptContainsLeanState:
    """The Lean state appears in the ChatML-rendered prompt."""

    def test_rendered_prompt_contains_lean_state(self):
        state = "case base\nn : ℕ\n⊢ n + 0 = n"
        tactic = "simp"
        record = make_record(state, tactic)
        messages = _standardize(record)
        rendered = _render(messages)

        assert state in rendered, (
            f"Lean state not found in rendered prompt.\n"
            f"State: {state!r}\nRendered:\n{rendered}"
        )

    def test_rendered_prompt_has_chatml_markers(self):
        record = make_record("⊢ True", "trivial")
        messages = _standardize(record)
        rendered = _render(messages)

        assert "<|im_start|>" in rendered
        assert "<|im_end|>" in rendered

    def test_rendered_prompt_contains_user_role(self):
        record = make_record("⊢ True", "trivial")
        messages = _standardize(record)
        rendered = _render(messages)
        assert "<|im_start|>user" in rendered

    def test_rendered_prompt_contains_assistant_role(self):
        record = make_record("⊢ True", "trivial")
        messages = _standardize(record)
        rendered = _render(messages)
        assert "<|im_start|>assistant" in rendered


# ── Test 3 ───────────────────────────────────────────────────────────────────

class TestTacticInAssistantSegmentOnly:
    """The tactic text appears after the assistant marker, not in the user segment."""

    def test_tactic_in_assistant_segment_only(self):
        state = "⊢ x = x"
        tactic = "rfl"
        record = make_record(state, tactic)
        messages = _standardize(record)
        rendered = _render(messages)

        # Split at assistant marker
        assert RESPONSE_PART in rendered, "Missing assistant marker in rendered output"
        before_assistant, after_assistant = rendered.split(RESPONSE_PART, 1)

        assert tactic in after_assistant, (
            f"Tactic {tactic!r} not found after assistant marker"
        )
        # Strip the trailing <|im_end|> from user segment to isolate user content
        # Tactic must not appear in the user portion of the rendered text
        assert tactic not in before_assistant, (
            f"Tactic {tactic!r} unexpectedly found in user segment:\n{before_assistant!r}"
        )

    def test_lean_state_in_user_segment(self):
        state = "h : P\n⊢ P"
        tactic = "exact h"
        record = make_record(state, tactic)
        messages = _standardize(record)
        rendered = _render(messages)

        before_assistant, _ = rendered.split(RESPONSE_PART, 1)
        assert state in before_assistant

    def test_user_preamble_in_user_segment(self):
        record = make_record("⊢ True", "trivial")
        messages = _standardize(record)
        rendered = _render(messages)

        before_assistant, _ = rendered.split(RESPONSE_PART, 1)
        assert "Lean 4 state" in before_assistant
        assert "Provide the next tactical step" in before_assistant


# ── Test 4 ───────────────────────────────────────────────────────────────────

class TestResponseOnlyMaskCoversUserTokens:
    """Simulate what train_on_responses_only does: user tokens → label -100.

    We work at the character level since we have no tokenizer.  The logic
    mirrors what train_on_responses_only does in the token stream:
      - Everything before (and including) the response marker is masked.
      - Everything after the response marker carries real labels.
    """

    def _compute_mask(self, rendered: str) -> list[int]:
        """Return a per-character label list (-100 = masked, 0 = unmasked)."""
        assert RESPONSE_PART in rendered, "No assistant marker found"
        # Position right after the response marker is where unmasking starts
        split_idx = rendered.index(RESPONSE_PART) + len(RESPONSE_PART)
        masked = [-100] * split_idx
        unmasked = [0] * (len(rendered) - split_idx)
        return masked + unmasked

    def test_response_only_mask_covers_user_tokens(self):
        state = "⊢ 1 + 1 = 2"
        tactic = "norm_num"
        record = make_record(state, tactic)
        messages = _standardize(record)
        rendered = _render(messages)
        mask = self._compute_mask(rendered)

        assert len(mask) == len(rendered)

        # Find where the state text sits in the rendered string
        state_start = rendered.index(state)
        state_end = state_start + len(state)

        # All characters of the state must be masked
        state_labels = mask[state_start:state_end]
        assert all(label == -100 for label in state_labels), (
            "User state characters are not fully masked"
        )

    def test_tactic_tokens_are_unmasked(self):
        state = "⊢ P"
        tactic = "exact hp"
        record = make_record(state, tactic)
        messages = _standardize(record)
        rendered = _render(messages)
        mask = self._compute_mask(rendered)

        # Find tactic in the rendered output (it appears in assistant segment)
        tactic_start = rendered.index(tactic)
        tactic_end = tactic_start + len(tactic)

        tactic_labels = mask[tactic_start:tactic_end]
        assert all(label == 0 for label in tactic_labels), (
            "Tactic characters are incorrectly masked"
        )

    def test_instruction_marker_itself_is_masked(self):
        record = make_record("⊢ True", "trivial")
        messages = _standardize(record)
        rendered = _render(messages)
        mask = self._compute_mask(rendered)

        # The user segment header (<|im_start|>user\n) must be masked
        instr_start = rendered.index(INSTRUCTION_PART)
        instr_end = instr_start + len(INSTRUCTION_PART)
        assert all(label == -100 for label in mask[instr_start:instr_end])

    def test_mask_length_equals_rendered_length(self):
        record = make_record("case h\n⊢ n = n", "rfl")
        messages = _standardize(record)
        rendered = _render(messages)
        mask = self._compute_mask(rendered)
        assert len(mask) == len(rendered)


# ── Test 5 ───────────────────────────────────────────────────────────────────

class TestTenSamplesRenderWithoutError:
    """10 distinct make_record() samples all render through the ChatML pipeline."""

    _SAMPLES = [
        ("⊢ n + 0 = n", "simp"),
        ("⊢ 0 + n = n", "simp [Nat.zero_add]"),
        ("h : P\n⊢ P", "exact h"),
        ("⊢ 1 + 1 = 2", "norm_num"),
        ("case inl\nh : P\n⊢ P ∨ Q", "exact Or.inl h"),
        ("⊢ ∀ n : ℕ, n + 0 = n", "intro n; simp"),
        ("f : α → β\ng : β → γ\n⊢ (g ∘ f) a = g (f a)", "rfl"),
        ("⊢ True", "trivial"),
        ("⊢ x = x", "rfl"),
        ("n : ℕ\n⊢ n < n + 1", "exact Nat.lt_succ_self n"),
    ]

    def test_ten_samples_render_without_error(self):
        results = []
        for state, tactic in self._SAMPLES:
            record = make_record(state, tactic)
            messages = _standardize(record)
            rendered = _render(messages)   # must not raise
            results.append(rendered)

        assert len(results) == 10

    def test_all_ten_contain_chatml_structure(self):
        for state, tactic in self._SAMPLES:
            record = make_record(state, tactic)
            messages = _standardize(record)
            rendered = _render(messages)
            assert INSTRUCTION_PART in rendered, (
                f"Missing user marker for state {state!r}"
            )
            assert RESPONSE_PART in rendered, (
                f"Missing assistant marker for state {state!r}"
            )

    def test_all_ten_state_and_tactic_present(self):
        for state, tactic in self._SAMPLES:
            record = make_record(state, tactic)
            messages = _standardize(record)
            rendered = _render(messages)
            assert state in rendered, f"State {state!r} missing from rendered output"
            assert tactic in rendered, f"Tactic {tactic!r} missing from rendered output"

    def test_all_ten_tactic_after_assistant_marker(self):
        for state, tactic in self._SAMPLES:
            record = make_record(state, tactic)
            messages = _standardize(record)
            rendered = _render(messages)
            _, after_assistant = rendered.split(RESPONSE_PART, 1)
            assert tactic in after_assistant, (
                f"Tactic {tactic!r} not found after assistant marker"
            )
