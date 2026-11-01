"""
RAG-Fusion Example
==================

Implements RAG-Fusion with reciprocal rank fusion and provider-agnostic chat model initialization.

| Section | Description |
|---------|-------------|
| Overview | Explanation of RAG-Fusion and the implementation |
| Setup   | Required environment variables |
| Usage   | How to run the example |

## Overview

RAG-Fusion is a retrieval-augmented generation technique that generates multiple
query variations, retrieves documents for each, and combines the results using
reciprocal rank fusion. This improves retrieval quality and answer accuracy.

## Setup

Set the following environment variables:

- ``CHAT_MODEL``: Chat model name (default: "gpt-4o-mini")
- ``CHAT_MODEL_PROVIDER``: Chat model provider (default: "openai")
- ``EMBEDDINGS_MODEL``: Embeddings model name (default: "text-embedding-3-small")
- ``EMBEDDINGS_PROVIDER``: Embeddings provider (default: "openai")

## Usage

Run this module directly to see a sample RAG-Fusion query:

    python examples/90_rag_fusion.py
"""

from __future__ import annotations

import os
from typing import List, Sequence, Tuple

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore


def generate_multi_queries(
    chat_model, query: str, num_queries: int = 3
) -> List[str]:
    """Generate multiple query variations using the chat model."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that generates diverse search queries. "
                "Generate {num_queries} different versions of the user's query to capture "
                "different aspects and phrasings. Return each query on a new line, "
                "without numbering or extra text.",
            ),
            ("human", "{query}"),
        ]
    )
    chain = prompt | chat_model
    response = chain.invoke({"query": query, "num_queries": num_queries})
    lines = response.content.strip().splitlines()
    queries = []
    for line in lines:
        line = line.strip()
        if line:
            # Remove leading numbering like "1." or "-"
            if line[0].isdigit() and "." in line[:3]:
                line = line.split(".", 1)[1].strip()
            elif line.startswith("- "):
                line = line[2:].strip()
            queries.append(line)
    return queries[:num_queries]


def reciprocal_rank_fusion(
    search_results: Sequence[Sequence[Document]], k: int = 60
) -> List[Tuple[Document, float]]:
    """Fuse multiple ranked document lists using reciprocal rank fusion."""
    fused_scores: dict[str, float] = {}
    documents: dict[str, Document] = {}

    for ranked_docs in search_results:
        for rank, doc in enumerate(ranked_docs):
            content = doc.page_content
            documents.setdefault(content, doc)
            fused_scores[content] = fused_scores.get(content, 0.0) + 1.0 / (k + rank + 1)

    # Sort by fused score descending
    sorted_docs = sorted(
        fused_scores.items(), key=lambda item: item[1], reverse=True
    )
    return [(documents[content], score) for content, score in sorted_docs]


def rag_fusion(
    chat_model,
    retriever,
    query: str,
    num_queries: int = 3,
    k: int = 60,
) -> str:
    """Run RAG-Fusion: generate queries, retrieve, fuse, and answer."""
    # Generate query variations
    queries = generate_multi_queries(chat_model, query, num_queries=num_queries)
    # Ensure the original query is included
    all_queries = [query] + queries

    # Retrieve documents for each query
    search_results = []
    for q in all_queries:
        docs = retriever.invoke(q)
        search_results.append(docs)

    # Fuse results
    fused_docs = reciprocal_rank_fusion(search_results, k=k)

    # Build context from fused documents
    context = "\n\n".join(doc.page_content for doc, _ in fused_docs)

    # Generate final answer
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer the user's question using only the provided context. "
                "If the answer is not in the context, say you don't know.",
            ),
            ("human", "Context:\n{context}\n\nQuestion: {query}"),
        ]
    )
    chain = answer_prompt | chat_model
    response = chain.invoke({"context": context, "query": query})
    return response.content


def main() -> None:
    """Run a sample RAG-Fusion query."""
    # Sample documents
    documents = [
        "The Eiffel Tower is located in Paris, France. It was completed in 1889.",
        "The Great Wall of China is a series of fortifications built across the northern borders of China.",
        "The Colosseum is an ancient amphitheater in Rome, Italy, built in 70-80 AD.",
        "Machu Picchu is a 15th-century Inca citadel located in the Andes Mountains of Peru.",
        "The Taj Mahal is an ivory-white marble mausoleum in Agra, India, built in 1632-1653.",
        "Chichen Itza is a Mayan city on the Yucatan Peninsula in Mexico, known for the pyramid El Castillo.",
    ]

    # Initialize provider-agnostic chat model and embeddings
    chat_model = init_chat_model(
        model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
        model_provider=os.getenv("CHAT_MODEL_PROVIDER", "openai"),
        temperature=0,
    )
    embeddings = init_embeddings(
        model=os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small"),
        model_provider=os.getenv("EMBEDDINGS_PROVIDER", "openai"),
    )

    # Create an in-memory vector store and retriever
    vector_store = InMemoryVectorStore.from_texts(
        documents, embedding=embeddings
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    query = "Which ancient structure is located in Italy?"
    answer = rag_fusion(chat_model, retriever, query, num_queries=3)
    print(f"Query: {query}\n")
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
