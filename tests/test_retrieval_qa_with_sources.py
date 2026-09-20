"""Tests for retrieval QA with sources.

| Test | Description |
|------|-------------|
| test_retrieval_qa_with_sources_returns_answer_and_sources | The chain returns an answer and parsed sources. |
| test_retrieval_qa_with_sources_answer_contains_source_references | The answer text includes the source URL. |
| test_retriever_called_with_query | The retriever is called with the user query. |
"""

from typing import List, Optional

from langchain.chains import RetrievalQAWithSourcesChain
from langchain_core.documents import Document
from langchain_core.language_models import LLM
from langchain_core.retrievers import BaseRetriever


class FakeLLM(LLM):
    responses: List[str]

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        return self.responses.pop(0)


class FakeRetriever(BaseRetriever):
    documents: List[Document]

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        return self.documents


class RecordingRetriever(FakeRetriever):
    last_query: str = ""

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        self.last_query = query
        return self.documents


def test_retrieval_qa_with_sources_returns_answer_and_sources():
    docs = [
        Document(
            page_content="The sky is blue.",
            metadata={"source": "https://example.com/sky"},
        ),
        Document(
            page_content="The ocean is blue.",
            metadata={"source": "https://example.com/ocean"},
        ),
    ]
    retriever = FakeRetriever(documents=docs)
    llm = FakeLLM(
        responses=[
            "The sky is blue.\nSOURCES: https://example.com/sky, https://example.com/ocean"
        ]
    )
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)

    result = chain("What color is the sky?")

    assert "answer" in result
    assert "sources" in result
    assert "https://example.com/sky" in result["sources"]
    assert "https://example.com/ocean" in result["sources"]


def test_retrieval_qa_with_sources_answer_contains_source_references():
    docs = [
        Document(
            page_content="The sky is blue.",
            metadata={"source": "https://example.com/sky"},
        )
    ]
    retriever = FakeRetriever(documents=docs)
    llm = FakeLLM(
        responses=[
            "The sky is blue. Source: https://example.com/sky\nSOURCES: https://example.com/sky"
        ]
    )
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)

    result = chain("What color is the sky?")

    assert "https://example.com/sky" in result["answer"]


def test_retriever_called_with_query():
    docs = [
        Document(
            page_content="The sky is blue.",
            metadata={"source": "https://example.com/sky"},
        )
    ]
    retriever = RecordingRetriever(documents=docs)
    llm = FakeLLM(
        responses=["The sky is blue.\nSOURCES: https://example.com/sky"]
    )
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)

    chain("What color is the sky?")

    assert retriever.last_query == "What color is the sky?"


if __name__ == "__main__":
    import os

    from langchain.chat_models import init_chat_model

    # Example: initialize a chat model without hardcoding credentials.
    # Set OPENAI_API_KEY or ANTHROPIC_API_KEY in your environment.
    llm = init_chat_model(
        model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
        provider=os.getenv("CHAT_MODEL_PROVIDER", "openai"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    print(f"Initialized model: {llm.__class__.__name__}")
    print("Run the tests with: pytest tests/test_retrieval_qa_with_sources.py")
