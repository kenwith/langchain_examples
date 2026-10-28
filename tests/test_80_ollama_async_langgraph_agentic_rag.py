"""Tests for the Ollama async LangGraph agentic RAG example.

| Test | Description |
|------|-------------|
| test_module_imports | Verifies the module can be imported. |
| test_graph_builder_runs_without_model | Verifies the graph builder executes without invoking a model. |
"""

import pytest
from ollama_async_langgraph_agentic_rag import build_graph


def test_module_imports():
    """Verify the module imports successfully."""
    import ollama_async_langgraph_agentic_rag as module
    assert hasattr(module, "build_graph")


def test_graph_builder_runs_without_model():
    """Verify the graph builder runs without invoking a model."""
    graph = build_graph()
    assert graph is not None


if __name__ == "__main__":
    pytest.main([__file__])
