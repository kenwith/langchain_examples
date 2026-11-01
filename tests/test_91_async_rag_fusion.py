"""Unit tests for the async retrieval helper with a mocked retriever."""

import asyncio
import os
from unittest.mock import AsyncMock

import pytest
from langchain.chat_models import init_chat_model

from langchain_examples.rag_fusion import async_retrieve


# ---------------------------------------------------------------------------
# 91. Async RAG Fusion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_retrieve_returns_documents():
    """The helper should return documents from the mocked retriever."""
    mock_retriever = AsyncMock()
    mock_retriever.ainvoke.return_value = ["doc1", "doc2"]

    result = await async_retrieve(mock_retriever, "test query")

    assert result == ["doc1", "doc2"]
    mock_retriever.ainvoke.assert_awaited_once_with("test query")


@pytest.mark.asyncio
async def test_async_retrieve_handles_empty_result():
    """The helper should return an empty list when the retriever returns nothing."""
    mock_retriever = AsyncMock()
    mock_retriever.ainvoke.return_value = []

    result = await async_retrieve(mock_retriever, "empty query")

    assert result == []
    mock_retriever.ainvoke.assert_awaited_once_with("empty query")


@pytest.mark.asyncio
async def test_async_retrieve_passes_query_correctly():
    """The helper should pass the exact query to the retriever."""
    mock_retriever = AsyncMock()
    mock_retriever.ainvoke.return_value = ["doc"]

    await async_retrieve(mock_retriever, "specific query")

    mock_retriever.ainvoke.assert_awaited_once_with("specific query")


@pytest.mark.asyncio
async def test_async_retrieve_propagates_retriever_exception():
    """Exceptions from the retriever should propagate to the caller."""
    mock_retriever = AsyncMock()
    mock_retriever.ainvoke.side_effect = RuntimeError("retriever failure")

    with pytest.raises(RuntimeError, match="retriever failure"):
        await async_retrieve(mock_retriever, "failing query")


def _demo_async_retrieve():
    """Demonstrate the async retrieval helper with a mocked retriever."""
    # Provider-agnostic model initialization (not used in this demo, shown for reference)
    model = init_chat_model("gpt-4o", model_provider="openai", api_key=os.getenv("OPENAI_API_KEY"))

    # A mocked retriever for demonstration purposes
    mock_retriever = AsyncMock()
    mock_retriever.ainvoke.return_value = [
        "Document 1 content",
        "Document 2 content",
    ]

    async def run():
        results = await async_retrieve(mock_retriever, "What is RAG fusion?")
        print("Retrieved documents:")
        for doc in results:
            print(f"- {doc}")

    asyncio.run(run())


if __name__ == "__main__":
    _demo_async_retrieve()
