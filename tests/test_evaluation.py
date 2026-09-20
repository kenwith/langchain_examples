"""Unit tests for the evaluation helper.

These tests verify the evaluation logic using sample predictions and
ground truth labels, without making any external model calls. The
external chat model is mocked so that only the evaluation logic is
exercised.

The evaluation helper is expected to use provider-agnostic
``init_chat_model`` to create a model, then ask it to judge whether
each prediction matches the ground truth. The helper returns a
dictionary with at least an ``accuracy`` key.
"""

from unittest.mock import MagicMock, patch

import pytest

from evaluation import evaluate_predictions


# | Test Case                 | Predictions          | Ground Truth         | Expected Accuracy |
# |---------------------------|----------------------|----------------------|-------------------|
# | All correct               | ["apple", "banana"]  | ["apple", "banana"]  | 1.0               |
# | Partial correct           | ["apple", "banana"]  | ["apple", "orange"]  | 0.5               |
# | Case-insensitive          | ["Apple", "Banana"]  | ["apple", "orange"]  | 0.5               |
# | Empty lists               | []                   | []                   | 0.0               |


def test_evaluate_predictions_all_correct():
    """Test that all-correct predictions yield 100% accuracy."""
    predictions = ["apple", "banana"]
    ground_truth = ["apple", "banana"]

    with patch("evaluation.init_chat_model") as mock_init:
        mock_model = MagicMock()
        mock_model.invoke.return_value = "correct"
        mock_init.return_value = mock_model

        result = evaluate_predictions(predictions, ground_truth)

    assert result["accuracy"] == 1.0


def test_evaluate_predictions_partial_correct():
    """Test that a mix of correct/incorrect predictions yields 50% accuracy."""
    predictions = ["apple", "banana"]
    ground_truth = ["apple", "orange"]

    with patch("evaluation.init_chat_model") as mock_init:
        mock_model = MagicMock()
        mock_model.invoke.side_effect = ["correct", "incorrect"]
        mock_init.return_value = mock_model

        result = evaluate_predictions(predictions, ground_truth)

    assert result["accuracy"] == 0.5


def test_evaluate_predictions_case_insensitive():
    """Test that the helper handles different capitalisation in model output."""
    predictions = ["Apple", "Banana"]
    ground_truth = ["apple", "orange"]

    with patch("evaluation.init_chat_model") as mock_init:
        mock_model = MagicMock()
        mock_model.invoke.side_effect = ["Correct", "Incorrect"]
        mock_init.return_value = mock_model

        result = evaluate_predictions(predictions, ground_truth)

    assert result["accuracy"] == 0.5


def test_evaluate_predictions_empty():
    """Test that empty inputs produce zero accuracy without errors."""
    with patch("evaluation.init_chat_model") as mock_init:
        mock_model = MagicMock()
        mock_init.return_value = mock_model

        result = evaluate_predictions([], [])

    assert result["accuracy"] == 0.0


def test_evaluate_predictions_mismatched_lengths():
    """Test that mismatched input lengths raise a ValueError."""
    with patch("evaluation.init_chat_model") as mock_init:
        mock_model = MagicMock()
        mock_init.return_value = mock_model

        with pytest.raises(ValueError):
            evaluate_predictions(["apple"], ["apple", "banana"])


if __name__ == "__main__":
    # Provider-agnostic model initialization is used by the evaluation helper.
    # In this test file we mock init_chat_model to avoid external API calls.
    # Run the tests with pytest, or invoke the test functions directly.
    pytest.main([__file__])
