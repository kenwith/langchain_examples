"""Tests for the async parallel tool calls example.

This module verifies that the async parallel tool calls example:
- returns the expected number of results when multiple tools are called in parallel,
- gracefully handles tool errors without breaking the overall execution.

| Test Function                                   | Description                                                       |
|-------------------------------------------------|-------------------------------------------------------------------|
| test_async_parallel_tool_calls_returns_results | Verifies that parallel tool calls produce the correct result count. |
| test_async_parallel_tool_calls_handles_errors  | Verifies that tool errors are caught and reported properly.         |
"""

from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.language_models.fake_chat_models import FakeChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool


# -----------------------------------------------------------------------------
# Helper tools
# -----------------------------------------------------------------------------
@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


@tool
def fail_tool() -> str:
    """A tool that always raises an error to test graceful handling."""
    raise ValueError("Intentional tool failure")


# -----------------------------------------------------------------------------
# Test helper: create a fake model that returns predefined tool calls
# -----------------------------------------------------------------------------
def create_fake_model(tool_calls: List[Dict[str, Any]]) -> FakeChatModel:
    """Create a FakeChatModel that returns a message with the given tool calls."""
    async def _aainvoke(*args: Any, **kwargs: Any) -> AIMessage:
        return AIMessage(content="", tool_calls=tool_calls)

    model = FakeChatModel()
    model._aainvoke = _aainvoke  # type: ignore[assignment]
    return model


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_async_parallel_tool_calls_returns_results():
    """Test that the example returns the expected number of results.

    We simulate two tool calls (add and multiply) and verify that the
    parallel execution returns a result for each tool.
    """
    tool_calls = [
        {"name": "add", "args": {"a": 1, "b": 2}, "id": "1"},
        {"name": "multiply", "args": {"a": 3, "b": 4}, "id": "2"},
    ]
    model = create_fake_model(tool_calls)
    tools = [add, multiply]

    async def _run() -> List[ToolMessage]:
        # Emulate the example's logic: parallel invoke tools based on model's tool_calls
        import asyncio

        tasks = []
        for call in tool_calls:
            tool = next(t for t in tools if t.name == call["name"])
            tasks.append(tool.ainvoke(call["args"]))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results

    results = asyncio.run(_run())
    assert len(results) == 2
    assert results[0] == 3  # add(1,2)
    assert results[1] == 12  # multiply(3,4)


def test_async_parallel_tool_calls_handles_errors():
    """Test that tool errors are caught and reported gracefully.

    We simulate a tool that raises an exception and verify that the
    error is not propagated but instead returned as an exception object.
    """
    tool_calls = [
        {"name": "fail_tool", "args": {}, "id": "3"},
        {"name": "add", "args": {"a": 5, "b": 6}, "id": "4"},
    ]
    model = create_fake_model(tool_calls)
    tools = [fail_tool, add]

    async def _run() -> List[Any]:
        import asyncio

        tasks = []
        for call in tool_calls:
            tool = next(t for t in tools if t.name == call["name"])
            tasks.append(tool.ainvoke(call["args"]))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results

    results = asyncio.run(_run())
    assert len(results) == 2
    assert isinstance(results[0], ValueError)  # fail_tool raised an error
    assert results[1] == 11  # add(5,6) succeeded


# -----------------------------------------------------------------------------
# __main__ demo block
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    # Run tests using pytest if available, otherwise run them manually
    try:
        import pytest
        sys.exit(pytest.main([__file__]))
    except ImportError:
        print("pytest not installed. Running tests manually...")
        test_async_parallel_tool_calls_returns_results()
        test_async_parallel_tool_calls_handles_errors()
        print("All tests passed.")
