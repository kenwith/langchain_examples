"""Tests for examples/56_ollama_evaluation.py.

These tests verify the evaluation helper functions provided in the
example, using mocked model and evaluator objects to avoid requiring
a live Ollama instance.
"""

import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure the examples directory is available for import.
EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
EXAMPLE_PATH = EXAMPLES_DIR / "56_ollama_evaluation.py"


def _load_example_module():
    """Load the example module from its file path."""
    spec = importlib.util.spec_from_file_location(
        "ollama_evaluation_example",
        EXAMPLE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def example_module():
    """Provide the example module as a test fixture."""
    return _load_example_module()


def test_get_llm_uses_environment_variable(example_module, monkeypatch):
    """The get_llm helper should read the model name from OLLAMA_MODEL."""
    monkeypatch.setenv("OLLAMA_MODEL", "ollama:test-model")
    fake_model = MagicMock()

    with patch.object(example_module, "init_chat_model", return_value=fake_model) as mock_init:
        model = example_module.get_llm()

    mock_init.assert_called_once_with(model="ollama:test-model", temperature=0)
    assert model is fake_model


def test_get_llm_uses_default_model(example_module, monkeypatch):
    """The get_llm helper should fall back to a default model name."""
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    fake_model = MagicMock()

    with patch.object(example_module, "init_chat_model", return_value=fake_model) as mock_init:
        model = example_module.get_llm()

    mock_init.assert_called_once()
    args, kwargs = mock_init.call_args
    assert "ollama" in kwargs.get("model", args[0] if args else "")
    assert model is fake_model


def test_evaluate_creates_evaluator(example_module):
    """The evaluate helper should create a criteria evaluator with the LLM."""
    fake_llm = MagicMock()
    fake_evaluator = MagicMock()
    fake_evaluator.evaluate_strings.return_value = {"score": 1}

    with (
        patch.object(example_module, "get_llm", return_value=fake_llm),
        patch.object(example_module, "load_evaluator", return_value=fake_evaluator) as mock_load,
        patch.object(fake_evaluator, "evaluate_strings", return_value={"score": 1}) as mock_eval,
    ):
        result = example_module.evaluate(
            question="What is the capital of France?",
            answer="Paris",
            criteria="correctness",
        )

    mock_load.assert_called_once_with(
        "criteria",
        llm=fake_llm,
        criteria="correctness",
    )
    mock_eval.assert_called_once_with(
        input="What is the capital of France?",
        prediction="Paris",
    )
    assert result == {"score": 1}


def test_evaluate_uses_default_criteria(example_module):
    """The evaluate helper should default to a standard criteria."""
    fake_llm = MagicMock()
    fake_evaluator = MagicMock()
    fake_evaluator.evaluate_strings.return_value = {"score": 1}

    with (
        patch.object(example_module, "get_llm", return_value=fake_llm),
        patch.object(example_module, "load_evaluator", return_value=fake_evaluator) as mock_load,
        patch.object(fake_evaluator, "evaluate_strings", return_value={"score": 1}),
    ):
        example_module.evaluate(
            question="What is the capital of France?",
            answer="Paris",
        )

    mock_load.assert_called_once()
    assert mock_load.call_args.kwargs["criteria"] is not None


def test_evaluate_returns_evaluator_result(example_module):
    """The evaluate helper should return the evaluator's result unchanged."""
    fake_llm = MagicMock()
    fake_evaluator = MagicMock()
    expected_result = {"reasoning": "correct", "score": 1}
    fake_evaluator.evaluate_strings.return_value = expected_result

    with (
        patch.object(example_module, "get_llm", return_value=fake_llm),
        patch.object(example_module, "load_evaluator", return_value=fake_evaluator),
    ):
        result = example_module.evaluate(
            question="What is the capital of France?",
            answer="Paris",
        )

    assert result == expected_result


if __name__ == "__main__":
    # Demonstrate the evaluation helpers with a mocked evaluator.
    example = _load_example_module()
    fake_evaluator = MagicMock()
    fake_evaluator.evaluate_strings.return_value = {"score": 1}
    print(
        example.evaluate(
            question="What is the capital of France?",
            answer="Paris",
        )
    )
