"""Example 24: RAG with Metadata Filtering.

This example demonstrates how to use metadata filtering during vector store
retrieval in a Retrieval-Augmented Generation (RAG) pipeline. It builds a
small in-memory vector store, adds documents with metadata, and then retrieves
relevant chunks based on a query while filtering by metadata (e.g., category).
The retrieved context is passed to a chat model to generate an answer.

The chat model is created with ``init_chat_model``, which is provider-agnostic.
Set the appropriate environment variable for your chosen provider (e.g.,
``OPENAI_API_KEY`` for OpenAI) before running this example.
"""

# ------------------------------------------------------------------
# | Example # | 24                                                |
# | Title     | Metadata Filtering RAG                            |
# | Purpose   | Show RAG with metadata filtering on vector store  |
# |           | retrieval.                                        |
# ------------------------------------------------------------------

import os

from langchain.chat_models import init_chat_model
from langchain_community.embeddings import FakeEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore


def create_documents():
    """Create a small set of documents with metadata."""
    return [
        Document(
            page_content="Apple is a technology company known for the iPhone and Mac.",
            metadata={"category": "technology", "year": 2023},
        ),
        Document(
            page_content="Microsoft develops software, including Windows and Office.",
            metadata={"category": "technology", "year": 2022},
        ),
        Document(
            page_content="JPMorgan Chase is a major financial institution.",
            metadata={"category": "finance", "year": 2023},
        ),
        Document(
            page_content="Goldman Sachs is an investment bank.",
            metadata={"category": "finance", "year": 2021},
        ),
        Document(
            page_content="Pfizer is a pharmaceutical company.",
            metadata={"category": "health", "year": 2022},
        ),
        Document(
            page_content="Moderna develops mRNA vaccines.",
            metadata={"category": "health", "year": 2023},
        ),
    ]


def build_vectorstore(documents):
    """Build an in-memory vector store from the documents."""
    embeddings = FakeEmbeddings(size=128)
    return InMemoryVectorStore.from_documents(documents, embedding=embeddings)


def build_metadata_filter(category=None, year=None):
    """Build a metadata filter dictionary from optional criteria."""
    filter_dict = {}
    if category is not None:
        filter_dict["category"] = category
    if year is not None:
        filter_dict["year"] = year
    return filter_dict


def retrieve_with_filter(vectorstore, query, filter_dict, k=3):
    """Retrieve documents using similarity search with metadata filtering."""
    return vectorstore.similarity_search(query, k=k, filter=filter_dict)


def generate_answer(query, context_documents):
    """Generate an answer using the chat model and retrieved context."""
    model = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0)
    context = "\n\n".join(doc.page_content for doc in context_documents)
    prompt = f"""Answer the question based solely on the provided context.

Context:
{context}

Question: {query}
"""
    response = model.invoke(prompt)
    return response.content


def main():
    """Run the metadata filtering RAG example."""
    print("Creating documents...")
    docs = create_documents()
    print(f"Created {len(docs)} documents.")

    print("Building vector store...")
    vectorstore = build_vectorstore(docs)

    query = "What does JPMorgan Chase do?"
    filter_dict = build_metadata_filter(category="finance")

    print("Retrieving without filter (first 2 results):")
    retrieved_no_filter = vectorstore.similarity_search(query, k=2)
    for doc in retrieved_no_filter:
        print(f"- {doc.page_content} (category: {doc.metadata['category']})")

    print(f"\nRetrieving with filter: {filter_dict}")
    retrieved = retrieve_with_filter(vectorstore, query, filter_dict, k=2)

    print(f"Retrieved {len(retrieved)} documents:")
    for doc in retrieved:
        print(f"- {doc.page_content} (category: {doc.metadata['category']})")

    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set. The chat model will not work.")
        return

    print("Generating answer...")
    answer = generate_answer(query, retrieved)
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
