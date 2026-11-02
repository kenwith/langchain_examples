"""Tests for metadata-filtered retrieval in RAG using an in-memory vector store."""

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import InMemoryVectorStore
from typing import List

from metadata_filtering_rag import build_metadata_filter


class FakeEmbeddings(Embeddings):
    """Deterministic fake embeddings for testing."""

    def __init__(self, dimension: int = 4):
        self.dimension = dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return [1.0, 0.0, 0.0, 0.0]


@pytest.fixture
def vector_store() -> InMemoryVectorStore:
    """Create an in-memory vector store with sample documents."""
    store = InMemoryVectorStore(embedding=FakeEmbeddings())
    documents = [
        Document(
            page_content="The sky is blue.",
            metadata={"category": "science", "language": "en"},
        ),
        Document(
            page_content="The ocean is blue.",
            metadata={"category": "science", "language": "en"},
        ),
        Document(
            page_content="Le ciel est bleu.",
            metadata={"category": "science", "language": "fr"},
        ),
        Document(
            page_content="The stock market is volatile.",
            metadata={"category": "finance", "language": "en"},
        ),
        Document(
            page_content="La bourse est volatile.",
            metadata={"category": "finance", "language": "fr"},
        ),
    ]
    store.add_documents(documents)
    return store


def test_filter_by_category(vector_store: InMemoryVectorStore) -> None:
    """Only documents with the requested category are returned."""
    results = vector_store.similarity_search("blue", filter={"category": "science"})
    assert results
    assert all(doc.metadata["category"] == "science" for doc in results)


def test_filter_by_language(vector_store: InMemoryVectorStore) -> None:
    """Only documents with the requested language are returned."""
    results = vector_store.similarity_search("blue", filter={"language": "fr"})
    assert results
    assert all(doc.metadata["language"] == "fr" for doc in results)


def test_filter_by_multiple_metadata(vector_store: InMemoryVectorStore) -> None:
    """Only documents matching all requested metadata are returned."""
    results = vector_store.similarity_search(
        "volatile",
        filter={"category": "finance", "language": "en"},
    )
    assert len(results) == 1
    assert results[0].metadata["category"] == "finance"
    assert results[0].metadata["language"] == "en"


def test_filter_no_match_returns_empty(vector_store: InMemoryVectorStore) -> None:
    """A filter that matches nothing returns an empty list."""
    results = vector_store.similarity_search("blue", filter={"category": "sports"})
    assert results == []


def test_retriever_metadata_filter(vector_store: InMemoryVectorStore) -> None:
    """The retriever applies metadata filters correctly."""
    retriever = vector_store.as_retriever(
        search_kwargs={"filter": {"category": "finance"}}
    )
    docs = retriever.invoke("volatile")
    assert docs
    assert all(doc.metadata["category"] == "finance" for doc in docs)


def test_build_metadata_filter_valid_single_condition() -> None:
    """A single valid key:value condition is parsed into a filter dict."""
    assert build_metadata_filter("category:science") == {"category": "science"}


def test_build_metadata_filter_valid_multiple_conditions() -> None:
    """Multiple space-separated key:value conditions are combined."""
    assert build_metadata_filter("category:science language:fr") == {
        "category": "science",
        "language": "fr",
    }


def test_build_metadata_filter_extra_whitespace_is_ignored() -> None:
    """Extra whitespace between conditions does not affect parsing."""
    assert build_metadata_filter("  category:science   language:fr  ") == {
        "category": "science",
        "language": "fr",
    }


def test_build_metadata_filter_empty_string() -> None:
    """An empty query string produces an empty filter dict."""
    assert build_metadata_filter("") == {}


def test_build_metadata_filter_invalid_missing_colon() -> None:
    """A condition without a colon is rejected."""
    with pytest.raises(ValueError):
        build_metadata_filter("category science")


def test_build_metadata_filter_invalid_empty_value() -> None:
    """A condition with an empty value is rejected."""
    with pytest.raises(ValueError):
        build_metadata_filter("category:")


def test_build_metadata_filter_invalid_empty_key() -> None:
    """A condition with an empty key is rejected."""
    with pytest.raises(ValueError):
        build_metadata_filter(":science")
