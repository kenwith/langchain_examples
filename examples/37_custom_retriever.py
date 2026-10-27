"""Example 37: Custom Retriever with InMemoryVectorStore.

This example demonstrates how to build custom retrievers from in-memory
document lists. Two retrievers are provided:

1. `InMemoryCustomRetriever` – a retriever that delegates similarity search
   to an `InMemoryVectorStore` with a deterministic embedding function.
2. `SimpleInMemoryRetriever` – a lightweight retriever that scores documents
   by simple word overlap with the query, without any vector store.

Both retrievers are subclasses of `BaseRetriever` and provide synchronous and
asynchronous retrieval methods. The `search_documents` helper formats the
retrieved documents into a numbered list.

Usage:
    The vector-store-based retriever can be created with
    `create_custom_retriever` and used directly or with a chat model.

    Example:
        retriever = create_custom_retriever()
        formatted = search_documents(retriever, "What is a retriever?")
        print(formatted)

    The simple keyword retriever can be created with `create_simple_retriever`
    and used in the same way.

    Example:
        retriever = create_simple_retriever()
        docs = retriever.invoke("What is a retriever?")
        context = "\\n\\n".join(doc.page_content for doc in docs)
        response = chat_model.invoke(
            f"Answer the question based on the context.\\n\\n"
            f"Context:\\n{context}\\n\\n"
            f"Question: What is a retriever?"
        )
        print(response.content)
"""

from collections import Counter
from typing import List

from langchain.chat_models import init_chat_model
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import InMemoryVectorStore
from pydantic import field_validator

import hashlib
import os


class SimpleEmbeddings(Embeddings):
    """Deterministic embeddings for demonstration purposes only.

    This embedding function is intentionally simple and deterministic. It is
    only meant to illustrate how to plug a custom embedding class into a
    vector store. Do not use this in production.
    """

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents.

        Args:
            texts: List of document texts to embed.

        Returns:
            List of embedding vectors, one per input text.
        """
        return [_embed_text(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """Embed a query text.

        Args:
            text: The query text to embed.

        Returns:
            The embedding vector for the query.
        """
        return _embed_text(text)


def _embed_text(text: str) -> List[float]:
    """Create a simple 32-dimensional vector from a text.

    This function uses MD5 hashes to produce a deterministic vector for a
    given text. It is not semantically meaningful, but it is sufficient for
    demonstrating a custom retriever.

    Args:
        text: The input text.

    Returns:
        A list of 32 floats in the range [0, 1].
    """
    vector = []
    for i in range(32):
        hash_input = f"{text}:{i}".encode()
        vector.append(int(hashlib.md5(hash_input).hexdigest(), 16) % 1000 / 1000.0)
    return vector


def create_documents() -> List[Document]:
    """Return a small list of in-memory documents.

    Returns:
        A list of `Document` objects with sample content about LangChain,
        retrievers, and vector stores.
    """
    return [
        Document(
            page_content="LangChain is a framework for developing applications "
            "powered by language models."
        ),
        Document(
            page_content="Retrievers are responsible for retrieving relevant "
            "documents from a knowledge base."
        ),
        Document(
            page_content="InMemoryVectorStore is a simple vector store that "
            "stores vectors in memory."
        ),
        Document(
            page_content="Custom retrievers can be built by subclassing "
            "BaseRetriever."
        ),
    ]


class InMemoryCustomRetriever(BaseRetriever):
    """Custom retriever backed by an in-memory vector store.

    This retriever wraps an `InMemoryVectorStore` and delegates the retrieval
    logic to its `similarity_search` and `asimilarity_search` methods. It is
    designed to be a minimal example of how to implement a custom retriever by
    subclassing `BaseRetriever`.

    The retriever is configured with a vector store and a number of documents
    to retrieve per query (`k`). The `invoke` method (inherited from
    `BaseRetriever`) calls the synchronous `_get_relevant_documents` method,
    which in turn calls `vector_store.similarity_search`. Similarly, `ainvoke`
    calls the asynchronous `_aget_relevant_documents` method, which calls
    `vector_store.asimilarity_search`.

    This class does not implement any additional filtering or post-processing
    of the retrieved documents; it is intentionally kept simple to illustrate
    the minimal interface required for a custom retriever. The `run_manager`
    arguments are required by the `BaseRetriever` interface and are provided by
    the framework for logging, tracing, and other callbacks. They are not used
    in this implementation.

    The constructor and assignment of attributes are handled by `BaseRetriever`
    through Pydantic. Input validation is performed to ensure that `k` is a
    positive integer and that queries passed to the retrieval methods are
    non-empty strings.

    Attributes:
        vector_store: The in-memory vector store used for similarity search.
        k: The number of documents to retrieve for each query. Defaults to 3.
            Must be a positive integer.
    """

    vector_store: InMemoryVectorStore
    k: int = 3

    @field_validator("k")
    @classmethod
    def validate_k(cls, value: int) -> int:
        """Validate that `k` is a positive integer.

        Args:
            value: The number of documents to retrieve.

        Returns:
            The validated value.

        Raises:
            ValueError: If `value` is less than or equal to zero.
        """
        if value <= 0:
            raise ValueError("k must be a positive integer")
        return value

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Retrieve documents relevant to the query.

        This method is called by `BaseRetriever.invoke` and should return a
        list of documents most relevant to the query. In this implementation,
        it simply delegates to the underlying vector store's similarity search.

        Args:
            query: The query string to search for. Must be a non-empty string.
            run_manager: Callback manager for the retriever run. Provided by
                the framework for logging, tracing, and other callbacks. This
                implementation does not use it, but it is required by the
                `BaseRetriever` interface.

        Returns:
            A list of `Document` objects sorted by relevance to the query.

        Raises:
            ValueError: If `query` is None, empty, or whitespace-only.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")
        return self.vector_store.similarity_search(query, k=self.k)

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Asynchronously retrieve documents relevant to the query.

        This method is called by `BaseRetriever.ainvoke` and should return a
        list of documents most relevant to the query. It delegates to the
        underlying vector store's asynchronous similarity search.

        Args:
            query: The query string to search for. Must be a non-empty string.
            run_manager: Async callback manager for the retriever run. Provided
                by the framework for logging, tracing, and other callbacks. This
                implementation does not use it, but it is required by the
                `BaseRetriever` interface.

        Returns:
            A list of `Document` objects sorted by relevance to the query.

        Raises:
            ValueError: If `query` is None, empty, or whitespace-only.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")
        return await self.vector_store.asimilarity_search(query, k=self.k)


class SimpleInMemoryRetriever(BaseRetriever):
    """Simple in-memory retriever based on word overlap.

    This retriever stores a list of documents directly in memory and scores
    each document by the number of query words that appear in its text. The
    top `k` documents with the highest scores are returned. It is provided as
    an even simpler alternative to `InMemoryCustomRetriever` and does not
    require embeddings or a vector store.

    The scoring function counts unique query words found in the document text.
    Documents with more matches are considered more relevant. Ties are broken
    by the original order of the document list.

    Attributes:
        documents: The list of documents to search over.
        k: The number of documents to retrieve for each query. Defaults to 3.
            Must be a positive integer.
    """

    documents: List[Document]
    k: int = 3

    @field_validator("k")
    @classmethod
    def validate_k(cls, value: int) -> int:
        """Validate that `k` is a positive integer.

        Args:
            value: The number of documents to retrieve.

        Returns:
            The validated value.

        Raises:
            ValueError: If `value` is less than or equal to zero.
        """
        if value <= 0:
            raise ValueError("k must be a positive integer")
        return value

    def _score_document(self, query_words: set, doc: Document) -> int:
        """Compute the overlap score between a query and a document.

        Args:
            query_words: A set of unique words from the query.
            doc: The document to score.

        Returns:
            The number of query words that appear in the document text.
        """
        doc_words = set(doc.page_content.lower().split())
        return len(query_words.intersection(doc_words))

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Retrieve documents relevant to the query.

        This method is called by `BaseRetriever.invoke` and returns the top
        `k` documents with the highest word-overlap scores.

        Args:
            query: The query string to search for. Must be a non-empty string.
            run_manager: Callback manager for the retriever run. Provided by
                the framework for logging, tracing, and other callbacks. This
                implementation does not use it, but it is required by the
                `BaseRetriever` interface.

        Returns:
            A list of `Document` objects sorted by relevance to the query.

        Raises:
            ValueError: If `query` is None, empty, or whitespace-only.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")
        query_words = set(query.lower().split())
        scored = [
            (self._score_document(query_words, doc), i, doc)
            for i, doc in enumerate(self.documents)
        ]
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [doc for _, _, doc in scored[: self.k]]

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Asynchronously retrieve documents relevant to the query.

        This method is called by `BaseRetriever.ainvoke` and returns the top
        `k` documents with the highest word-overlap scores. Since the scoring
        is purely in-memory and synchronous, this method simply calls the
        synchronous implementation.

        Args:
            query: The query string to search for. Must be a non-empty string.
            run_manager: Async callback manager for the retriever run. Provided
                by the framework for logging, tracing, and other callbacks. This
                implementation does not use it, but it is required by the
                `BaseRetriever` interface.

        Returns:
            A list of `Document` objects sorted by relevance to the query.

        Raises:
            ValueError: If `query` is None, empty, or whitespace-only.
        """
        return self._get_relevant_documents(
            query, run_manager=run_manager
        )


def create_custom_retriever() -> InMemoryCustomRetriever:
    """Create a custom retriever from an in-memory document list.

    Returns:
        An `InMemoryCustomRetriever` instance configured with the sample
        documents and the deterministic embedding function.
    """
    documents = create_documents()
    vector_store = InMemoryVectorStore.from_documents(
        documents, embedding=SimpleEmbeddings()
    )
    return InMemoryCustomRetriever(vector_store=vector_store)


def create_simple_retriever() -> SimpleInMemoryRetriever:
    """Create a simple in-memory retriever from the sample documents.

    Returns:
        A `SimpleInMemoryRetriever` instance configured with the sample
        documents and default `k` value.
    """
    return SimpleInMemoryRetriever(documents=create_documents())


def search_documents(retriever: BaseRetriever, query: str) -> str:
    """Retrieve documents for a query and format them as a numbered list.

    Args:
        retriever: The retriever to use for document retrieval.
        query: The query string to search for.

    Returns:
        A formatted string with each retrieved document on a separate line,
        prefixed by its position.
    """
    docs = retriever.invoke(query)
    lines = [f"{i + 1}. {doc.page_content}" for i, doc in enumerate(docs)]
    return "\n".join(lines)


def main() -> None:
    """Run a simple retrieval-augmented generation demo.

    The demo retrieves documents for a sample query using the vector-store
    based retriever, builds a context block from the retrieved documents, and
    asks a chat model to answer the query based on that context.
    """
    retriever = create_custom_retriever()
    query = "What is a retriever?"

    print("Retrieving documents for query:", query)
    retrieved_docs = retriever.invoke(query)
    print(f"Retrieved {len(retrieved_docs)} documents.\n")

    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    chat_model = init_chat_model(os.getenv("CHAT_MODEL", "openai:gpt-4o-mini"))
    response = chat_model.invoke(
        f"Answer the question based on the context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}"
    )
    print("Answer:", response.content)


if __name__ == "__main__":
    main()
