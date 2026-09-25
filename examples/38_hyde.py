"""
Example 38: HyDE (Hypothetical Document Embeddings)

This example demonstrates how to use hypothetical document embeddings
for retrieval-augmented generation (RAG). A chat model generates a
hypothetical document from the query, and that document is embedded
and used to retrieve relevant documents from a corpus. The retrieved
documents are then passed to the chat model to generate a grounded answer.

Uses init_chat_model and init_embeddings for provider-agnostic model
initialization. Requires API keys for the chosen providers.
"""

import os

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings


def generate_hypothetical_document(query: str, chat_model) -> str:
    """Generate a hypothetical document that would answer the query."""
    prompt = (
        "Write a short, factual passage that would be a good search result "
        "for the following query:\n\n"
        f"{query}\n\n"
        "Passage:"
    )
    response = chat_model.invoke(prompt)
    return response.content.strip()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute the cosine similarity between two vectors."""
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def retrieve_relevant_documents(
    query_embedding: list[float],
    document_embeddings: list[list[float]],
    k: int = 2,
) -> list[int]:
    """Return the indices of the top-k most similar documents."""
    similarities = [
        cosine_similarity(query_embedding, doc_embedding)
        for doc_embedding in document_embeddings
    ]
    top_indices = sorted(
        range(len(similarities)),
        key=lambda i: similarities[i],
        reverse=True,
    )[:k]
    return top_indices


def answer_question(query: str, context: str, chat_model) -> str:
    """Answer the query using the provided context."""
    prompt = (
        "Answer the question using only the following context:\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
    response = chat_model.invoke(prompt)
    return response.content.strip()


def main() -> None:
    """Run the HyDE retrieval-augmented generation example."""
    # Initialize provider-agnostic models.
    # Set OPENAI_API_KEY or other provider keys as needed.
    chat_model = init_chat_model(
        os.getenv("CHAT_MODEL", "gpt-4o-mini"),
        model_provider=os.getenv("CHAT_MODEL_PROVIDER", "openai"),
    )
    embeddings = init_embeddings(
        os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        model_provider=os.getenv("EMBEDDING_PROVIDER", "openai"),
    )

    # A small corpus of documents.
    documents = [
        "LangChain is a framework for developing applications powered by language models.",
        "RAG combines retrieval and generation to produce grounded answers.",
        "HyDE uses a hypothetical document to improve retrieval accuracy.",
        "FAISS is a library for efficient similarity search of dense vectors.",
        "Embeddings convert text into numerical vectors for semantic search.",
    ]

    # Embed the documents once.
    document_embeddings = embeddings.embed_documents(documents)

    # The user query.
    query = "What is HyDE and how does it improve RAG?"

    # Generate a hypothetical document from the query.
    hypothetical_doc = generate_hypothetical_document(query, chat_model)
    print("Hypothetical document:\n", hypothetical_doc, "\n")

    # Embed the hypothetical document and retrieve relevant documents.
    hypothetical_embedding = embeddings.embed_query(hypothetical_doc)
    top_indices = retrieve_relevant_documents(
        hypothetical_embedding, document_embeddings, k=2
    )
    retrieved_documents = [documents[i] for i in top_indices]
    print("Retrieved documents:")
    for i, doc in zip(top_indices, retrieved_documents):
        print(f"  [{i}] {doc}")
    print()

    # Answer the query using the retrieved documents as context.
    context = "\n\n".join(retrieved_documents)
    answer = answer_question(query, context, chat_model)
    print("Answer:\n", answer)


if __name__ == "__main__":
    main()
