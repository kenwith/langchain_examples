"""Unit tests for the Ollama async LangGraph retrieval grader example.

This test suite verifies the relevance classification helper and the graph
construction logic used in the retrieval grader example. All LLM calls are
mocked to avoid external API dependencies.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

# The example module is expected to live under either langchain_examples.examples
# or examples. Try both import paths to remain compatible with different repo
# layouts.
try:
    from langchain_examples.examples.ollama_async_langgraph_retrieval_grader import (
        classify_relevance,
        create_graph,
    )
except ImportError:
    from examples.ollama_async_langgraph_retrieval_grader import (
        classify_relevance,
        create_graph,
    )


# ---------------------------------------------------------------------------
# Test: Relevance classification helper
# ---------------------------------------------------------------------------

def test_classify_relevance_yes():
    """The helper should return 'yes' when the LLM classifies a document as relevant."""
    with patch(
        "langchain_examples.examples.ollama_async_langgraph_retrieval_grader.init_chat_model"
    ) as mock_init:
        mock_model = MagicMock()
        mock_model.invoke.return_value = "yes"
        mock_init.return_value = mock_model

        result = classify_relevance(
            "What is LangChain?",
            "LangChain is a framework for building applications with LLMs.",
        )
        assert result == "yes"


def test_classify_relevance_no():
    """The helper should return 'no' when the LLM classifies a document as irrelevant."""
    with patch(
        "langchain_examples.examples.ollama_async_langgraph_retrieval_grader.init_chat_model"
    ) as mock_init:
        mock_model = MagicMock()
        mock_model.invoke.return_value = "no"
        mock_init.return_value = mock_model

        result = classify_relevance(
            "What is LangChain?",
            "The weather is sunny today.",
        )
        assert result == "no"


def test_classify_relevance_uses_llm_prompt():
    """The helper should pass the query and document to the LLM."""
    with patch(
        "langchain_examples.examples.ollama_async_langgraph_retrieval_grader.init_chat_model"
    ) as mock_init:
        mock_model = MagicMock()
        mock_model.invoke.return_value = "yes"
        mock_init.return_value = mock_model

        classify_relevance("query", "document")

        # Verify the LLM was called with a prompt containing both inputs.
        call_args = mock_model.invoke.call_args[0][0]
        assert "query" in call_args
        assert "document" in call_args


# ---------------------------------------------------------------------------
# Test: Graph construction
# ---------------------------------------------------------------------------

def test_create_graph_returns_compiled_graph():
    """The graph builder should return a compiled graph with an invoke method."""
    with patch(
        "langchain_examples.examples.ollama_async_langgraph_retrieval_grader.init_chat_model"
    ) as mock_init:
        mock_model = MagicMock()
        mock_init.return_value = mock_model

        graph = create_graph()
        assert hasattr(graph, "invoke")


def test_create_graph_contains_expected_nodes():
    """The graph should contain the classification node and a routing node."""
    with patch(
        "langchain_examples.examples.ollama_async_langgraph_retrieval_grader.init_chat_model"
    ) as mock_init:
        mock_model = MagicMock()
        mock_init.return_value = mock_model

        graph = create_graph()
        graph_info = graph.get_graph()

        # Collect node names from the graph representation.
        node_names = {node.id for node in graph_info.nodes}

        # The exact node names depend on the example implementation, but
        # typically include 'classify_relevance' and something like
        # 'should_continue'. Adjust these assertions to match the actual
        # implementation if needed.
        assert "classify_relevance" in node_names
        assert "should_continue" in node_names


def test_create_graph_uses_provider_agnostic_model():
    """The graph builder should use init_chat_model without hardcoded provider settings."""
    with patch(
        "langchain_examples.examples.ollama_async_langgraph_retrieval_grader.init_chat_model"
    ) as mock_init:
        mock_model = MagicMock()
        mock_init.return_value = mock_model

        create_graph()

        # Ensure init_chat_model was called with provider/model from env vars.
        kwargs = mock_init.call_args.kwargs
        assert "provider" in kwargs
        assert "model" in kwargs
        assert kwargs["provider"] == os.getenv("CHAT_MODEL_PROVIDER")
        assert kwargs["model"] == os.getenv("CHAT_MODEL")


# ---------------------------------------------------------------------------
# Demo block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__])
