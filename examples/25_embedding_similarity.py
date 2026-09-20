"""Example 25: Embedding Similarity.

This example demonstrates how to initialize a provider-agnostic embeddings model
(using init_embeddings_model, the embeddings counterpart to init_chat_model)
and rank a small set of documents by cosine similarity to a query.
"""

import math
import os

from langchain.embeddings import init_embeddings_model


# ---------------------------------------------------------------------------
# Example 25: Embedding Similarity
# ---------------------------------------------------------------------------


def cosine_similarity(vec_a, vec_b):
    """Compute the cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def rank_documents(query, documents, embeddings):
    """Rank documents by cosine similarity to the query."""
    query_vec = embeddings.embed_query(query)
    doc_vecs = embeddings.embed_documents(documents)

    scored = []
    for i, doc_vec in enumerate(doc_vecs):
        score = cosine_similarity(query_vec, doc_vec)
        scored.append((i, score, documents[i]))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def main():
    """Run the embedding similarity example."""
    # A small set of documents to rank.
    documents = [
        "Pasta is a staple food of Italian cuisine.",
        "Dogs are loyal companions and great pets.",
        "The Eiffel Tower is located in Paris, France.",
        "Quantum mechanics is a fundamental theory in physics.",
        "Chocolate is made from cocoa beans and is often sweet.",
    ]

    query = "Tell me about Italian food."

    # Initialize a provider-agnostic embeddings model.
    # Set the model via the EMBEDDINGS_MODEL environment variable, or use the default.
    # Make sure to set the appropriate API key for your provider (e.g., OPENAI_API_KEY).
    model = os.getenv("EMBEDDINGS_MODEL", "openai:text-embedding-3-small")
    embeddings = init_embeddings_model(model)

    results = rank_documents(query, documents, embeddings)

    print(f"Query: {query}\n")
    print("Ranked documents:")
    for i, score, doc in results:
        print(f"{i + 1}. Similarity: {score:.4f} - {doc}")


if __name__ == "__main__":
    main()
