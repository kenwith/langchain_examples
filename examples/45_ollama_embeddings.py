"""Example 45: Local Embeddings with Ollama and Similarity Search.

This example demonstrates how to generate local embeddings using Ollama
through LangChain's provider-agnostic `init_embeddings` API (the same
pattern as `init_chat_model`), and then perform a simple similarity
search over a set of documents.

Prerequisites:
- Install LangChain with Ollama support: `pip install langchain langchain-ollama`
- Have Ollama running locally with an embedding model pulled, e.g.:
  `ollama pull nomic-embed-text`
- Set the `OLLAMA_BASE_URL` environment variable if your Ollama server
  is not at the default `http://localhost:11434`.
"""

from __future__ import annotations

import os

import numpy as np
from langchain.embeddings import init_embeddings


def get_embeddings_model() -> object:
    """Return a configured Ollama embeddings model."""
    return init_embeddings(
        "ollama",
        model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )


def embed_documents(embeddings_model: object, texts: list[str]) -> list[list[float]]:
    """Embed a list of documents and return the vectors."""
    return embeddings_model.embed_documents(texts)


def embed_query(embeddings_model: object, query: str) -> list[float]:
    """Embed a query and return the vector."""
    return embeddings_model.embed_query(query)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def similarity_search(
    embeddings_model: object, query: str, documents: list[str], top_k: int = 2
) -> list[tuple[str, float]]:
    """Return the top_k most similar documents to the query."""
    query_embedding = embed_query(embeddings_model, query)
    document_embeddings = embed_documents(embeddings_model, documents)

    scored_docs = [
        (doc, cosine_similarity(query_embedding, doc_emb))
        for doc, doc_emb in zip(documents, document_embeddings)
    ]
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return scored_docs[:top_k]


def main() -> None:
    """Run the example."""
    print("Initializing Ollama embeddings model...")
    embeddings_model = get_embeddings_model()

    documents = [
        "LangChain is a framework for developing applications powered by language models.",
        "Ollama allows you to run large language models locally on your machine.",
        "Embeddings convert text into numerical vectors that capture semantic meaning.",
        "The sky is blue because of Rayleigh scattering.",
    ]

    query = "What is Ollama?"

    print(f"Query: {query}\n")
    print("Documents:")
    for doc in documents:
        print(f" - {doc}")

    print("\nPerforming similarity search...")
    results = similarity_search(embeddings_model, query, documents, top_k=2)

    print("\nTop matches:")
    for doc, score in results:
        print(f"  Score: {score:.4f} | {doc}")


if __name__ == "__main__":
    main()
