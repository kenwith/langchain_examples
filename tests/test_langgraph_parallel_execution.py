"""Test the parallel execution example graph.

This test verifies that the build_graph() function compiles and
exposes the expected nodes and edges for parallel execution.
"""

import pytest
from langchain_examples.parallel_execution import build_graph


def test_build_graph_compiles():
    """Build the graph and ensure it compiles."""
    app = build_graph()
    # StateGraph is compiled on demand; if not compiled yet, do so now.
    if hasattr(app, "compile"):
        app = app.compile()
    graph = app.get_graph()
    assert graph.nodes is not None
    assert graph.edges is not None


def test_parallel_nodes_and_edges():
    """Verify the graph has parallel branches from start to end."""
    app = build_graph()
    if hasattr(app, "compile"):
        app = app.compile()
    graph = app.get_graph()

    nodes = set(graph.nodes.keys())
    edges = set(graph.edges)

    # Core nodes
    assert "start" in nodes
    assert "end" in nodes

    # There must be at least two intermediate nodes for parallel execution.
    intermediate = nodes - {"start", "end"}
    assert len(intermediate) >= 2, "Expected at least two parallel nodes"

    # Each intermediate node must be connected from start and to end.
    for node in intermediate:
        assert ("start", node) in edges, f"Missing edge from start to {node}"
        assert (node, "end") in edges, f"Missing edge from {node} to end"


if __name__ == "__main__":
    # Simple demo to show the graph structure.
    app = build_graph()
    if hasattr(app, "compile"):
        app = app.compile()
    graph = app.get_graph()
    print("Nodes:", list(graph.nodes.keys()))
    print("Edges:", list(graph.edges))
