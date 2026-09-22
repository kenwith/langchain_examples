"""
# Tests for LLM-as-Judge Example

This test module verifies two critical pieces of the `llm_as_judge` example:

1. **Judge score parsing** — the ability to extract a numeric score
   from the raw LLM response, regardless of formatting.
2. **Rubric-based evaluation** — the structure of the result returned
   by the evaluation function when given a mock LLM.

| Test Function               | Purpose                                                        |
|-----------------------------|----------------------------------------------------------------|
| `test_parse_score`          | Validates score extraction from plain, labeled, and JSON        |
|                             | responses.                                                      |
| `test_parse_score_invalid`  | Ensures malformed responses raise a `ValueError`.              |
| `test_evaluate_with_rubric` | Confirms the evaluation result contains `score` and `feedback`. |

Run the tests:

```bash
pytest tests/test_llm_as_judge.py
```

For a live demo (requires API access), set `RUN_LLM_AS_JUDGE_DEMO=1`
and the appropriate provider/model environment variables.
"""

import os
import pytest
from langchain.chat_models import init_chat_model  # noqa: F401 (used in demo)

try:
    from llm_as_judge import parse_score, evaluate_with_rubric
except ImportError:
    from examples.llm_as_judge import parse_score, evaluate_with_rubric


class MockLLM:
    """A minimal LLM stub that returns a fixed string for any prompt."""

    def __init__(self, response):
        self.response = response

    def invoke(self, prompt):
        return self.response


def test_parse_score():
    """Score should be extracted from various common response formats."""
    assert parse_score("Score: 4") == 4
    assert parse_score('{"score": 3, "feedback": "ok"}') == 3
    assert parse_score("4") == 4
    assert parse_score("Rating: 5/5") == 5


def test_parse_score_invalid():
    """A response with no identifiable score should raise ValueError."""
    with pytest.raises(ValueError):
        parse_score("The answer is unknown.")


def test_evaluate_with_rubric():
    """The evaluation function should return a structured result."""
    fake_llm = MockLLM('{"score": 5, "feedback": "Excellent work"}')
    result = evaluate_with_rubric(
        llm=fake_llm,
        problem="What is 2+2?",
        answer="4",
        rubric="Correctness (1-5)"
    )
    assert isinstance(result, dict)
    assert result["score"] == 5
    assert result["feedback"] == "Excellent work"


if __name__ == "__main__":
    # ---------- Demo (optional) ----------
    # Set RUN_LLM_AS_JUDGE_DEMO=1 and provide model credentials to
    # actually call a live LLM. Otherwise only a mock is shown.
    if os.getenv("RUN_LLM_AS_JUDGE_DEMO", "0") == "1":
        model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        provider = os.getenv("LLM_PROVIDER", "openai")
        llm = init_chat_model(model, provider=provider)
        sample = evaluate_with_rubric(
            llm=llm,
            problem="What is the capital of France?",
            answer="Paris",
            rubric="Accuracy (1-5)"
        )
        print(sample)
    else:
        # A simple local demonstration using the mock class.
        mock = MockLLM('{"score": 4, "feedback": "Good but could be better"}')
        result = evaluate_with_rubric(
            llm=mock,
            problem="What is 2+2?",
            answer="4",
            rubric="Correctness (1-5)"
        )
        print("Mock evaluation result:", result)
