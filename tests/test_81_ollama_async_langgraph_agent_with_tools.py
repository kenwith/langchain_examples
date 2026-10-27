"""Unit tests for example 81: Ollama async LangGraph agent with tools.

| Example | Description |
|---------|-------------|
| 81      | Ollama async LangGraph agent with tools |
"""

import asyncio
import importlib.util
import os
from unittest.mock import MagicMock, patch

import pytest

EXAMPLE_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "examples",
    "81_ollama_async_langgraph_agent_with_tools.py",
)


def load_example():
    """Load the example module from the examples directory."""
    spec = importlib.util.spec_from_file_location("example_81", EXAMPLE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tool_invocation_and_response():
    """Verify that the agent invokes a tool and returns its response."""
    example = load_example()

    tool = example.get_weather
    tool_calls = []

    def recording_invoke(city):
        tool_calls.append(city)
        return f"Weather in {city}: sunny"

    tool.invoke = recording_invoke

    class FakeAgent:
        def __init__(self, model, tools):
            self.model = model
            self.tools = tools

        async def ainvoke(self, messages):
            result = self.tools[0].invoke("Paris")
            return {"messages": [{"role": "assistant", "content": result}]}
