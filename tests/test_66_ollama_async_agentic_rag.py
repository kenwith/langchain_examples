"""Test for the Ollama async agentic RAG example.

This test runs the example's main function with mocked retrieval and chat
calls to avoid external dependencies.
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from examples.ollama_async_agentic_rag import main as example_main


@pytest.mark.asyncio
async def test_ollama_async_agentic_rag():
    """Run the example with mocked retrieval and chat calls."""
    # Mock the retriever to return sample documents.
    mock_retriever = MagicMock()
    mock_retriever.ainvoke = AsyncMock(
        return_value=[
            MagicMock(page_content="Sample document content")
        ]
    )

    # Mock the chat model to return a canned response.
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value="Mocked RAG response")

    # Patch the example's retriever and chat model.
    with patch("examples.ollama_async_agentic_rag.retriever", mock_retriever), \
         patch("examples.ollama_async_agentic_rag.llm", mock_llm):
        await example_main()

    # Verify that retrieval and chat calls were made.
    mock_retriever.ainvoke.assert_awaited_once()
    mock_llm.ainvoke.assert_awaited()


if __name__ == "__main__":
    pytest.main([__file__])
