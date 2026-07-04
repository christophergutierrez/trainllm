import textwrap

from code_verify import clean_completion, verify_humaneval


def test_clean_completion_extracts_fenced_python():
    completion = """```python
def add_one(x):
    return x + 1
```
"""
    assert clean_completion(completion) == "def add_one(x):\n    return x + 1"


def test_clean_completion_truncates_trailing_fence_explanation():
    completion = """def add_one(x):
    return x + 1
```
This is the solution.
"""
    assert clean_completion(completion) == "def add_one(x):\n    return x + 1"


def test_verify_humaneval_appends_missing_check_call():
    prompt = "def add_one(x):\n"
    completion = "    return x + 1"
    test_code = textwrap.dedent(
        """
        def check(candidate):
            assert candidate(1) == 2
            assert candidate(-1) == 0
        """
    )

    result = verify_humaneval(prompt, completion, test_code, entry_point="add_one")

    assert result.passed


def test_verify_humaneval_rejects_wrong_solution_with_appended_check_call():
    prompt = "def add_one(x):\n"
    completion = "    return x"
    test_code = textwrap.dedent(
        """
        def check(candidate):
            assert candidate(1) == 2
        """
    )

    result = verify_humaneval(prompt, completion, test_code, entry_point="add_one")

    assert not result.passed
