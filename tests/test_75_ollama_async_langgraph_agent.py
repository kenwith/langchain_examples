"""Test for the Ollama async LangGraph agent example.

This test verifies that the example module imports cleanly and that its
core functions can be invoked without requiring external API keys.
"""

import sys
from pathlib import Path

# Add the project root to sys.path to import the example module.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.ollama_async_langgraph_agent import build_graph


def test_build_graph_returns_compiled_graph():
    """Verify the build_graph function returns a graph without external calls."""
    graph = build_graph()
    assert graph is not None
    # A compiled LangGraph usually has an invoke method.
    assert hasattr(graph, "invoke")


if __name__ == "__main__":
    test_build_graph_returns_compiled_graph()
    print("All tests passed.")
