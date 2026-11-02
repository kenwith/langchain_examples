"""Async tests for the async Ollama RAG-Fusion example."""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# The example module is assumed to be at examples.async_ollama_rag_fusion.
# Tests mock the async retrievers and the chat model to avoid external calls.
from examples.async_ollama_rag_fusion import (
    generate_queries,
    generate_answer,
    reciprocal_rank_fusion,
    retrieve_and_fuse,
)


# Test: Reciprocal Rank Fusion
def test_reciprocal_rank_fusion():
    """Verify that reciprocal rank fusion correctly combines ranked lists."""
    ranked_docs = [
        ["doc1", "doc2", "doc3"],
        ["doc2", "doc1", "doc4"],
        ["doc3", "doc4", "doc1"],
    ]
    fused = reciprocal_rank_fusion(ranked_docs, k=60)
    # doc1 appears in all three lists at positions 0,1,2 -> scores 1/61 + 1/62 + 1/63
    # doc2 appears in first two lists at positions 1,0 -> 1/62 + 1/61
    # doc3 appears in first and third lists at positions 2,0 -> 1/63 + 1/61
    # doc4 appears in second and third lists at positions 2,1 -> 1/63 + 1/62
    assert fused[0] == "doc1"
    assert fused[1] == "doc2"
    assert fused[2] == "doc3"
    assert fused[3] == "doc4"


# Test: Query Generation (async)
@pytest.mark.asyncio
async def test_generate_queries_async():
    """Verify that generate_queries returns multiple query variations."""
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = "What is RAG?\nHow does RAG work?\nExplain RAG fusion."
    with patch("examples.async_ollama_rag_fusion.init_chat_model", return_value=mock_llm):
        queries = await generate_queries("What is RAG?")
    assert len(queries) == 3
    assert queries[0] == "What is RAG?"
    assert queries[1] == "How does RAG work?"
    assert queries[2] == "Explain RAG fusion."


# Test: Async Retrieval and Fusion
@pytest.mark.asyncio
async def test_retrieve_and_fuse_async():
    """Verify that the async retrievers are called and results are fused."""
    mock_retriever = AsyncMock()
    mock_retriever.ainvoke.side_effect = [
        ["docA", "docB"],
        ["docB", "docC"],
        ["docC", "docA"],
    ]
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = "Query1\nQuery2\nQuery3"

    with patch("examples.async_ollama_rag_fusion.init_chat_model", return_value=mock_llm):
        with patch("examples.async_ollama_rag_fusion.retriever", mock_retriever):
            fused_docs = await retrieve_and_fuse("What is RAG?")

    assert mock_retriever.ainvoke.await_count == 3
    assert fused_docs[0] == "docA"
    assert fused_docs[1] == "docB"
    assert fused_docs[2] == "docC"


# Test: Answer Generation (async)
@pytest.mark.asyncio
async def test_generate_answer_async():
    """Verify that the answer generation uses the fused context."""
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = "RAG is a framework for retrieval-augmented generation."

    with patch("examples.async_ollama_rag_fusion.init_chat_model", return_value=mock_llm):
        answer = await generate_answer("What is RAG?", ["doc1", "doc2"])

    assert "RAG" in answer
    mock_llm.ainvoke.assert_awaited_once()


# Test: Provider-Agnostic Model Initialization
def test_init_chat_model_provider_agnostic():
    """Verify that init_chat_model is used without hardcoding a provider."""
    with patch("examples.async_ollama_rag_fusion.init_chat_model") as mock_init:
        from examples.async_ollama_rag_fusion import get_model
        get_model()
        mock_init.assert_called_once()
        # Ensure no provider-specific arguments are passed
        call_kwargs = mock_init.call_args.kwargs
        assert "provider" not in call_kwargs
        assert "model" in call_kwargs


# Demo block
if __name__ == "__main__":
    async def demo():
        """Run a quick demonstration of the async RAG-Fusion pipeline."""
        # Use mocked retrievers and model to avoid network calls
        mock_retriever = AsyncMock()
        mock_retriever.ainvoke.side_effect = [
            ["LangChain is a framework.", "RAG combines retrieval and generation."],
            ["RAG fusion improves answer quality.", "Async support is built-in."],
            ["Ollama runs models locally.", "RAG-Fusion uses multiple queries."],
        ]
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = "RAG-Fusion is a technique that combines multiple retrieval queries."

        with patch("examples.async_ollama_rag_fusion.init_chat_model", return_value=mock_llm):
            with patch("examples.async_ollama_rag_fusion.retriever", mock_retriever):
                question = "What is RAG-Fusion?"
                queries = await generate_queries(question)
                fused_docs = await retrieve_and_fuse(question)
                answer = await generate_answer(question, fused_docs)
                print("Queries:", queries)
                print("Fused docs:", fused_docs)
                print("Answer:", answer)

    asyncio.run(demo())
