"""Tests for the async_adaptive_rag example."""

import importlib


def test_async_adaptive_rag_import_and_nodes():
    """Verify the example imports and the compiled graph exposes expected nodes."""
    # Import the example module, trying both common package layouts.
    try:
        example = importlib.import_module("langchain_examples.async_adaptive_rag")
    except ImportError:
        example = importlib.import_module("examples.async_adaptive_rag")

    # Locate a function that builds the compiled graph.
    create_func = None
    for name in ("create_workflow", "build_graph", "create_graph"):
        if hasattr(example, name):
            create_func = getattr(example, name)
            break

    if create_func is not None:
        graph = create_func()
    else:
        # Fall back to a module-level compiled graph attribute.
        for attr in ("graph", "app", "workflow"):
            candidate = getattr(example, attr, None)
            if candidate is not None and hasattr(candidate, "get_graph"):
                graph = candidate
                break
        else:
            raise AssertionError(
                "No workflow creation function or compiled graph found in the example."
            )

    # Verify that the expected nodes are present in the compiled graph.
    node_names = set(graph.get_graph().nodes.keys())
    expected_nodes = {
        "retrieve",
        "grade_documents",
        "decide_to_generate",
        "generate",
        "rewrite",
    }
    missing = expected_nodes - node_names
    assert not missing, f"Missing expected nodes: {missing}"

    print("Async adaptive RAG example imports and exposes expected nodes.")


if __name__ == "__main__":
    test_async_adaptive_rag_import_and_nodes()
