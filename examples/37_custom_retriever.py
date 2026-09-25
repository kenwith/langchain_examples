"""Example 37: Custom Retriever with InMemoryVectorStore.

This example demonstrates how to build a custom retriever from an in-memory
document list using `InMemoryVectorStore`, and how to use it with a chat model.

The custom retriever is implemented as a subclass of `BaseRetriever` and
delegates the actual similarity search to an `InMemoryVectorStore`. This
keeps the retriever simple while still allowing custom behaviour to be added
later.
"""

from typing import List

from langchain.chat_models import init_chat_model
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import InMemoryVectorStore

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
    """Custom retriever backed by an InMemoryVectorStore.

    This retriever stores a reference to an `InMemoryVectorStore` and uses its
    `similarity_search` method to find relevant documents. The number of
    documents to return is controlled by the `k` field.
    """

    vector_store: InMemoryVectorStore
    k: int = 3

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Retrieve documents relevant to the query.

        Args:
            query: The query string.
            run_manager: Callback manager for the retriever run. Used for
                logging and tracing; not used in this simple implementation.

        Returns:
            A list of `Document` objects sorted by relevance.
        """
        return self.vector_store.similarity_search(query, k=self.k)


def create_custom_retriever() -> BaseRetriever:
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


def main() -> None:
    """Run a simple retrieval-augmented generation demo.

    The demo retrieves documents for a sample query, builds a context block
    from the retrieved documents, and asks a chat model to answer the query
    based on that context.
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
