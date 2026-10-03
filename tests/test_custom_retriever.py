"""Tests for the custom retriever example.

This module demonstrates a custom retriever that supports optional LLM-based query
expansion. The tests use a deterministic fake LLM so they never require API keys.

| Test | Description |
|------|-------------|
| test_retriever_returns_matching_documents | Ensures documents containing query terms are returned. |
| test_retriever_uses_llm_query_expansion | Ensures the retriever can use an LLM to expand the query. |
| test_retriever_returns_empty_when_no_match | Ensures an empty list is returned for non-matching queries. |
"""

import pytest
from types import SimpleNamespace
from typing import List, Optional

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever


class CustomRetriever(BaseRetriever):
    """A simple custom retriever with optional LLM query expansion.

    Attributes:
        documents: List of documents to search.
        llm: Optional chat model used to expand the query before matching.
    """

    documents: List[Document]
    llm: Optional[BaseChatModel] = None

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """Return documents whose content contains any query keyword."""
        if self.llm is not None:
            expanded = self.llm.invoke(f"Expand this query: {query}")
            query_text = expanded.content if hasattr(expanded, "content") else str(expanded)
            keywords = query_text.split()
        else:
            keywords = query.split()

        return [
            doc
            for doc in self.documents
            if any(keyword.lower() in doc.page_content.lower() for keyword in keywords)
        ]


class FakeLLM:
    """A fake LLM that returns a fixed response for tests."""

    def invoke(self, prompt: str):
        return SimpleNamespace(content="langchain")


def test_retriever_returns_matching_documents():
    docs = [
        Document(page_content="LangChain is a framework for LLM apps."),
        Document(page_content="Python is a programming language."),
    ]
    retriever = CustomRetriever(documents=docs)

    result = retriever.invoke("tell me about langchain")

    assert len(result) == 1
    assert result[0].page_content == docs[0].page_content


def test_retriever_uses_llm_query_expansion():
    docs = [
        Document(page_content="LangChain is a framework for LLM apps."),
        Document(page_content="Python is a programming language."),
    ]
    fake_llm = FakeLLM()
    retriever = CustomRetriever(documents=docs, llm=fake_llm)

    result = retriever.invoke("what is langchain?")

    assert len(result) == 1
    assert result[0].page_content == docs[0].page_content


def test_retriever_returns_empty_when_no_match():
    docs = [Document(page_content="Java is a programming language.")]
    retriever = CustomRetriever(documents=docs)

    result = retriever.invoke("rust")

    assert result == []


def main():
    """Run a quick demo of the custom retriever with a real LLM (if configured)."""
    try:
        llm = init_chat_model()
    except Exception:
        print("Skipping demo: no chat model configured.")
        return

    retriever = CustomRetriever(
        documents=[
            Document(page_content="LangChain is a framework for LLM apps."),
            Document(page_content="Python is a programming language."),
        ],
        llm=llm,
    )

    print(retriever.invoke("langchain"))


if __name__ == "__main__":
    main()
