"""
Test suite for example 84: Adaptive RAG.
"""

import os
import pytest
from unittest.mock import patch

from langchain.chat_models import init_chat_model
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

# Import the example module (adjust path as needed)
try:
    from examples.example_84_adaptive_rag import (
        build_retrieval_grader,
        build_adaptive_graph,
    )
except ImportError:
    # Fallback for different directory structure
    from example_84_adaptive_rag import (
        build_retrieval_grader,
        build_adaptive_graph,
    )


# Test: Retrieval Grader
# -----------------------
def test_retrieval_grader():
    """Validate the retrieval grader returns a binary relevance score."""
    fake_model = FakeMessagesListChatModel(
        responses=[
            AIMessage(content='{"score": "yes"}'),
            AIMessage(content='{"score": "no"}'),
        ]
    )
    grader = build_retrieval_grader(fake_model)
    assert grader.invoke({"question": "What is LangChain?", "document": "LangChain is a framework."}) == {"score": "yes"}
    assert grader.invoke({"question": "What is LangChain?", "document": "The weather is sunny."}) == {"score": "no"}


# Test: Adaptive Graph Construction
# ---------------------------------
def test_adaptive_graph():
    """Validate the adaptive graph is constructed with expected nodes and edges."""
    fake_model = FakeMessagesListChatModel(responses=[AIMessage(content="")])
    graph = build_adaptive_graph(fake_model)
    # Check that the graph has the expected nodes
    nodes = set(graph.get_graph().nodes.keys())
    assert "retrieve" in nodes
    assert "grade_documents" in nodes
    assert "generate" in nodes
    # Check that the graph is compiled
    assert graph is not None


# Example of using init_chat_model in a provider-agnostic way (not executed in tests)
def _example_init_chat_model():
    """Demonstrate provider-agnostic model initialization."""
    model = init_chat_model(os.getenv("CHAT_MODEL", "gpt-4o-mini"))
    return model


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__]))
