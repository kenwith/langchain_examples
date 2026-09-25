"""Example 37: Custom Retriever with InMemoryVectorStore.

This example demonstrates how to build a custom retriever from an in-memory
document list using `InMemoryVectorStore`, and how to use it with a chat model.
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
    """Deterministic embeddings for demonstration purposes only."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [_embed_text(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return _embed_text(text)


def _embed_text(text: str) -> List[float]:
    """Create a simple 32-dimensional vector from a text."""
    vector = []
    for i in range(32):
        hash_input = f"{text}:{i}".encode()
        vector.append(int(hashlib.md5(hash_input).hexdigest(), 16) % 1000 / 1000.0)
    return vector


def create_documents() -> List[Document]:
    """Return a small list of in-memory documents."""
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
    """Custom retriever backed by an InMemoryVectorStore."""

    vector_store: InMemoryVectorStore
    k: int = 3

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return self.vector_store.similarity_search(query, k=self.k)


def create_custom_retriever() -> BaseRetriever:
    """Create a custom retriever from an in-memory document list."""
    documents = create_documents()
    vector_store = InMemoryVectorStore.from_documents(
        documents, embedding=SimpleEmbeddings()
    )
    return InMemoryCustomRetriever(vector_store=vector_store)


def main() -> None:
    """Run a simple retrieval-augmented generation demo."""
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
