"""Smoke tests for the async agent with memory module.

This test suite verifies that the async agent graph builder is callable and
that the provider-agnostic model initialization helper is available.

| Test | Description |
|------|-------------|
| test_async_agent_graph_builder_is_callable | Asserts that the graph builder function is callable. |
"""

import os

from langchain.chat_models import init_chat_model

from langchain_examples.async_agent_with_memory import build_async_agent_graph


def test_async_agent_graph_builder_is_callable():
    """Verify the graph builder function is callable."""
    assert callable(build_async_agent_graph)


if __name__ == "__main__":
    test_async_agent_graph_builder_is_callable()

    model = init_chat_model(os.getenv("CHAT_MODEL", "openai:gpt-4o-mini"), temperature=0)
    print(f"Initialized model: {model.__class__.__name__}")
