"""Example 24: RAG with Metadata Filtering.

This example demonstrates how to use metadata filtering during vector store
retrieval in a Retrieval-Augmented Generation (RAG) pipeline. It builds a
small in-memory vector store, adds documents with metadata, and then retrieves
relevant chunks based on a query while filtering by metadata (e.g., category).
The retrieved context is passed to a chat model to generate an answer.

The chat model is created with ``init_chat_model``, which is provider-agnostic.
Set the appropriate environment variable for your chosen provider (e.g.,
``OPENAI_API_KEY`` for OpenAI) before running this example.

This example also showcases the ``build_filtered_retriever`` function, which
centralizes metadata filtering by constructing a retriever with the appropriate
search kwargs. Additionally, the ``build_filter_from_query`` helper constructs
metadata filters from simple query keywords, making it easy to derive filters
directly from natural language input.
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


def build_filter_from_query(query):
    """Build a metadata filter dict from simple keywords in a query.

    This helper inspects the query for known keywords and maps them to
    metadata fields. It is intentionally simple and can be extended with
    more sophisticated parsing.
    """
    filter_dict = {}
    keyword_map = {
        "apple": {"category": "technology"},
        "iphone": {"category": "technology"},
        "microsoft": {"category": "technology"},
        "software": {"category": "technology"},
        "jpmorgan": {"category": "finance"},
        "chase": {"category": "finance"},
        "goldman": {"category": "finance"},
        "bank": {"category": "finance"},
        "pfizer": {"category": "health"},
        "moderna": {"category": "health"},
        "vaccine": {"category": "health"},
        "pharma": {"category": "health"},
    }
    query_lower = query.lower()
    for keyword, metadata in keyword_map.items():
        if keyword in query_lower:
            filter_dict.update(metadata)

    # Detect a four-digit year in the query and add it to the filter.
    for token in query_lower.split():
        if token.isdigit() and len(token) == 4:
            filter_dict["year"] = int(token)

    return filter_dict


def build_filtered_retriever(vectorstore, filter_dict, k=3):
    """Build a retriever that applies metadata filtering.

    This centralizes the metadata filter configuration. The returned retriever
    will use the given filter and top-k value for all its searches.
    """
    return vectorstore.as_retriever(search_kwargs={"filter": filter_dict, "k": k})


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
    filter_dict = build_filter_from_query(query)

    print("Retrieving without filter (first 2 results):")
    retrieved_no_filter = vectorstore.similarity_search(query, k=2)
    for doc in retrieved_no_filter:
        print(f"- {doc.page_content} (category: {doc.metadata['category']})")

    print(f"\nBuilding filtered retriever with filter: {filter_dict}")
    filtered_retriever = build_filtered_retriever(vectorstore, filter_dict, k=2)

    print("Retrieving with filter using the retriever:")
    retrieved = filtered_retriever.invoke(query)

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
