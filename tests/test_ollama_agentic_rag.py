"""Tests for the Ollama agentic RAG example."""

import importlib
import sys
from pathlib import Path

import pytest

pytest.importorskip("ollama")
pytest.importorskip("langchain_ollama")

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))


def test_agent_graph_compiles():
    """Import the example and verify the agent graph compiles."""
    module = importlib.import_module("ollama_agentic_rag")
    graph = (
        getattr(module, "graph", None)
        or getattr(module, "agentic_rag_graph", None)
        or getattr(module, "app", None)
    )
    assert graph is not None

    if hasattr(graph, "compile"):
        compiled = graph.compile()
        assert compiled is not None
        assert hasattr(compiled, "invoke")
    else:
        assert hasattr(graph, "invoke")
