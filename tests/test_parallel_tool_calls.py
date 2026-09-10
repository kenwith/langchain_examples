"""Unit tests for the parallel tool-call example."""

from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage

from examples.parallel_tool_calls import run_parallel_tool_calls


def create_mock_model(response):
    """Create a mock model that returns a given response when invoked."""
    mock_model = Mock()
    bound_model = Mock()
    bound_model.invoke.return_value = response
    mock_model.bind_tools.return_value = bound_model
    return mock_model


def create_mock_tool(name, return_value):
    """Create a mock tool with a fixed return value."""
    tool = Mock()
    tool.name = name
    tool.invoke.return_value = return_value
    return tool


# ---------------------------------------------------------------------------
# Test: test_parallel_tool_calls_executes_tools_in_parallel
# Description: Verify that multiple tool calls are executed in parallel.
# Expected: Both tools are called with the correct arguments.
# ---------------------------------------------------------------------------
def test_parallel_tool_calls_executes_tools_in_parallel():
    response = AIMessage(
        content="",
        tool_calls=[
            {"name": "add", "args": {"a": 1, "b": 2}, "id": "call_1"},
            {"name": "multiply", "args": {"a": 3, "b": 4}, "id": "call_2"},
        ],
    )
    mock_model = create_mock_model(response)
    mock_add = create_mock_tool("add", 3)
    mock_multiply = create_mock_tool("multiply", 12)

    with patch("examples.parallel_tool_calls.init_chat_model", return_value=mock_model), patch(
        "examples.parallel_tool_calls.tools", [mock_add, mock_multiply]
    ):
        result = run_parallel_tool_calls("What is 1+2 and 3*4?")

    mock_add.invoke.assert_called_once_with({"a": 1, "b": 2})
    mock_multiply.invoke.assert_called_once_with({"a": 3, "b": 4})
    assert result is not None
    assert len(result) == 2


# ---------------------------------------------------------------------------
# Test: test_parallel_tool_calls_no_tool_calls
# Description: Verify that no tools are called when the model returns no tool calls.
# Expected: The original model response is returned unchanged.
# ---------------------------------------------------------------------------
def test_parallel_tool_calls_no_tool_calls():
    response = AIMessage(content="I don't know", tool_calls=[])
    mock_model = create_mock_model(response)
    mock_add = create_mock_tool("add", 3)
    mock_multiply = create_mock_tool("multiply", 12)

    with patch("examples.parallel_tool_calls.init_chat_model", return_value=mock_model), patch(
        "examples.parallel_tool_calls.tools", [mock_add, mock_multiply]
    ):
        result = run_parallel_tool_calls("Hello")

    mock_add.invoke.assert_not_called()
    mock_multiply.invoke.assert_not_called()
    assert result == response


# ---------------------------------------------------------------------------
# Test: test_parallel_tool_calls_handles_tool_error
# Description: Verify that a tool error is captured and returned as a ToolMessage.
# Expected: The error message is included in the result.
# ---------------------------------------------------------------------------
def test_parallel_tool_calls_handles_tool_error():
    response = AIMessage(
        content="",
        tool_calls=[
            {"name": "add", "args": {"a": 1, "b": 2}, "id": "call_1"},
        ],
    )
    mock_model = create_mock_model(response)
    mock_add = Mock()
    mock_add.name = "add"
    mock_add.invoke.side_effect = ValueError("boom")

    with patch("examples.parallel_tool_calls.init_chat_model", return_value=mock_model), patch(
        "examples.parallel_tool_calls.tools", [mock_add]
    ):
        result = run_parallel_tool_calls("What is 1+2?")

    mock_add.invoke.assert_called_once_with({"a": 1, "b": 2})
    assert isinstance(result, list)
    assert len(result) == 1
    assert "boom" in result[0].content


if __name__ == "__main__":
    pytest.main([__file__])
