"""Tests for the ollama_parallel_tool_calls example.

## Overview

This test module verifies the behavior of the `ollama_parallel_tool_calls`
example by mocking the chat model. It ensures the example's `run` function
executes without error and correctly interacts with the model.

## Test matrix

| Test name | Description |
|-----------|-------------|
| test_run_with_mocked_chat_model | Checks that `run` calls `init_chat_model`, `bind_tools`, and `invoke` as expected. |
"""

import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import ollama_parallel_tool_calls


def test_run_with_mocked_chat_model():
    """Verify the example runs with a mocked chat model."""
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.tool_calls = [
        {
            "name": "add",
            "args": {"x": 1, "y": 2},
            "id": "call_1",
        },
        {
            "name": "multiply",
            "args": {"x": 3, "y": 4},
            "id": "call_2",
        },
    ]
    mock_model.invoke.return_value = mock_response
    mock_model.bind_tools.return_value = mock_model

    with patch(
        "ollama_parallel_tool_calls.init_chat_model",
        return_value=mock_model,
    ) as mock_init:
        result = ollama_parallel_tool_calls.run()

    mock_init.assert_called_once()
    mock_model.bind_tools.assert_called_once()
    mock_model.invoke.assert_called_once()
    assert result is mock_response


if __name__ == "__main__":
    test_run_with_mocked_chat_model()
    print("Test passed.")
