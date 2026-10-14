"""Tests for the Ollama async LangGraph agentic RAG example."""

import asyncio
import importlib.util
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

# Path to the example module
EXAMPLE_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "examples",
    "ollama_async_langgraph_agentic_rag.py",
)


def _load_example_module():
    """Load the example module from the examples directory."""
    spec = importlib.util.spec_from_file_location(
        "ollama_async_langgraph_agentic_rag", EXAMPLE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_main(module):
    """Run the module's main function, handling async definitions."""
    if asyncio.iscoroutinefunction(module.main):
        asyncio.run(module.main())
    else:
        module.main()


def test_example_imports():
    """Verify the example module can be imported and exposes a main function."""
    module = _load_example_module()
    assert callable(module.main)


def test_example_main_executes(monkeypatch):
    """Verify the example main function executes with mocked dependencies."""
    module = _load_example_module()

    # Mock the chat model initialization
    mock_llm = MagicMock()
    monkeypatch.setattr(module, "init_chat_model", lambda *args, **kwargs: mock_llm)

    # Mock the embeddings
    mock_embeddings = MagicMock()
    monkeypatch.setattr(
        module, "OllamaEmbeddings", lambda *args, **kwargs: mock_embeddings
    )

    # Mock the vector store
    mock_vectorstore = MagicMock()
    monkeypatch.setattr(
        module, "InMemoryVectorStore", lambda *args, **kwargs: mock_vectorstore
    )

    # Mock the agent graph and its invocation
    mock_graph = MagicMock()
    mock_graph.invoke = AsyncMock(return_value={"answer": "Mocked answer"})
    monkeypatch.setattr(module, "create_agent", lambda *args, **kwargs: mock_graph)

    # Run the main function
    _run_main(module)

    # Verify the mocked graph was used
    mock_graph.invoke.assert_awaited_once()


if __name__ == "__main__":
    pytest.main([__file__])
