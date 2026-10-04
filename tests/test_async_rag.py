"""Tests for the async RAG example.

This test verifies that the async RAG chain builds correctly and returns a
string answer when retrieval is mocked.
"""

import asyncio
import os
from unittest.mock import AsyncMock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


def build_chain(retriever):
    """Build an async RAG chain using a provider-agnostic chat model."""
    from langchain.chat_models import init_chat_model

    model = init_chat_model(
        os.getenv("CHAT_MODEL", "openai:gpt-4o-mini"),
        temperature=0,
    )
    prompt = ChatPromptTemplate.from_template(
        "Answer the question based on the context.\n\n"
        "Context: {context}\n\n"
        "Question: {question}"
    )
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )


# ---------------------------------------------------------------------------
# Test: Async RAG returns a string answer
# ---------------------------------------------------------------------------
async def _run_async_rag_chain():
    """Run the async RAG chain with mocked retrieval and model."""
    retriever = AsyncMock()
    retriever.ainvoke.return_value = [
        Document(
            page_content=(
                "LangChain is a framework for developing applications "
                "powered by language models."
            )
        )
    ]

    mock_model = AsyncMock()
    mock_model.ainvoke.return_value = AIMessage(
        content="LangChain is a framework."
    )

    with patch("langchain.chat_models.init_chat_model", return_value=mock_model):
        chain = build_chain(retriever)
        result = await chain.ainvoke("What is LangChain?")

    retriever.ainvoke.assert_awaited_once_with("What is LangChain?")
    mock_model.ainvoke.assert_awaited_once()
    return result


def test_async_rag_returns_string():
    """Test that the async RAG chain returns a string answer."""
    result = asyncio.run(_run_async_rag_chain())
    assert isinstance(result, str)
    assert "LangChain" in result


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_async_rag_returns_string()
    print("Test passed.")
