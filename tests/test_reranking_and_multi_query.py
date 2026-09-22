"""Tests for the reranking and multi-query retrieval examples.

These tests mock the LLM and vector store to validate the example
pipelines without requiring external API calls.
"""

import pytest
from unittest.mock import Mock, patch

pytest.importorskip("examples.reranking")
pytest.importorskip("examples.multi_query")

import examples.reranking as reranking
import examples.multi_query as multi_query


def _make_fake_llm(responses):
    """Return a Mock LLM that returns the given responses."""
    llm = Mock()
    llm.invoke.side_effect = responses
    return llm


def _make_fake_vectorstore(documents):
    """Return a Mock vector store with a retriever returning documents."""
    retriever = Mock()
    retriever.invoke.return_value = documents
    vectorstore = Mock()
    vectorstore.as_retriever.return_value = retriever
    return vectorstore


# ---------------------------------------------------------------------------
# Test: Reranking Example
# ---------------------------------------------------------------------------

def test_reranking_example():
    """Test the reranking example with a mocked LLM and vector store."""
    fake_llm = _make_fake_llm(["relevant", "irrelevant"])
    fake_vectorstore = _make_fake_vectorstore(
        [
            Mock(page_content="doc1", metadata={}),
            Mock(page_content="doc2", metadata={}),
        ]
    )

    with patch.object(reranking, "init_chat_model", return_value=fake_llm):
        with patch.object(reranking, "vectorstore", fake_vectorstore):
            result = reranking.main()

    assert result is not None
    assert fake_vectorstore.as_retriever.called
    assert fake_llm.invoke.called


# ---------------------------------------------------------------------------
# Test: Multi-Query Example
# ---------------------------------------------------------------------------

def test_multi_query_example():
    """Test the multi-query example with a mocked LLM and vector store."""
    fake_llm = _make_fake_llm(["query1\nquery2", "answer"])
    fake_vectorstore = _make_fake_vectorstore(
        [
            Mock(page_content="doc1", metadata={}),
            Mock(page_content="doc2", metadata={}),
        ]
    )

    with patch.object(multi_query, "init_chat_model", return_value=fake_llm):
        with patch.object(multi_query, "vectorstore", fake_vectorstore):
            result = multi_query.main()

    assert result is not None
    assert fake_vectorstore.as_retriever.called
    assert fake_llm.invoke.called


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__]))
