"""
Example 31: Multi-Query Retrieval
================================

| Example | Description |
|---------|-------------|
| 31 | Multi-Query Retrieval |

This example demonstrates how to use multi-query retrieval to improve recall
by generating multiple queries from an original question.

Multi-query retrieval expands a single user question into several related
queries using an LLM. This helps overcome ambiguous phrasing, covers different
perspectives on the same information need, and increases the chance that
relevant documents are retrieved from the vector store. The final result set
is a union of documents retrieved by each generated query, which can
substantially improve recall compared to a single query.

This example also normalizes the generated queries by removing duplicates
before retrieval, reducing redundant vector store calls while preserving the
recall benefits of multi-query retrieval.
"""

import os

from langchain.chat_models import init_chat_model
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Sample documents for the in-memory vector store
DOCUMENTS = [
    Document(page_content="LangChain is a framework for developing applications powered by language models."),
    Document(page_content="Multi-query retrieval generates multiple search queries from a single user question."),
    Document(page_content="Retrieval augmented generation (RAG) combines retrieval with generation."),
    Document(page_content="FAISS is a library for efficient similarity search and clustering of dense vectors."),
    Document(page_content="The MultiQueryRetriever uses an LLM to generate variations of the original query."),
    Document(page_content="Improving recall means retrieving more relevant documents even if the original query is ambiguous."),
]


def create_vectorstore() -> FAISS:
    """Create and return an in-memory FAISS vector store from sample documents."""
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(DOCUMENTS, embeddings)


def generate_multi_queries(retriever: MultiQueryRetriever, question: str) -> list[str]:
    """Generate multiple query variations from an original question.

    Args:
        retriever: A MultiQueryRetriever configured with an LLM.
        question: The original user question.

    Returns:
        A list of generated query strings.
    """
    return retriever.generate_queries(question)


def deduplicate_queries(queries: list[str]) -> list[str]:
    """Remove duplicate queries while preserving original order.

    Normalization trims whitespace and lowercases the query for comparison,
    so variations that differ only by case or surrounding spaces are treated
    as duplicates.

    Args:
        queries: A list of query strings.

    Returns:
        A list of unique query strings in their original form.
    """
    seen = set()
    unique_queries = []
    for query in queries:
        normalized = query.strip().lower()
        if normalized not in seen:
            seen.add(normalized)
            unique_queries.append(query)
    return unique_queries


def retrieve_with_deduplicated_queries(retriever, queries: list[str]) -> list[Document]:
    """Retrieve documents for each unique query and combine the results.

    The base retriever is called once per unique query, avoiding redundant
    vector store calls for duplicate queries. Documents are also deduplicated
    by their page content to keep the final result set clean.

    Args:
        retriever: The base retriever used to search the vector store.
        queries: A list of unique query strings.

    Returns:
        A list of deduplicated documents retrieved across all queries.
    """
    docs = []
    seen_docs = set()
    for query in queries:
        for doc in retriever.invoke(query):
            if doc.page_content not in seen_docs:
                seen_docs.add(doc.page_content)
                docs.append(doc)
    return docs


def print_generated_queries(queries: list[str]) -> None:
    """Print the generated queries."""
    print("Generated queries:")
    for i, query in enumerate(queries, 1):
        print(f"{i}. {query}")
    print()


def run_example() -> None:
    """Run the multi-query retrieval example."""
    # Provider-agnostic chat model initialization
    llm = init_chat_model(os.getenv("MODEL", "gpt-4o"), temperature=0)

    vectorstore = create_vectorstore()

    # Build a MultiQueryRetriever from the base retriever and the LLM.
    # The base retriever is configured with search_kwargs={"k": 2} to limit
    # the number of documents returned per generated query. The LLM is used
    # by MultiQueryRetriever to produce multiple query variations from the
    # original question, which helps improve recall by covering different
    # phrasings or perspectives of the same information need.
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
    )

    question = "What is multi-query retrieval and how does it improve recall?"
    print(f"Question: {question}\n")

    # Generate query variations and normalize them by removing duplicates.
    # This reduces redundant vector store calls during retrieval.
    queries = generate_multi_queries(retriever, question)
    print_generated_queries(queries)

    unique_queries = deduplicate_queries(queries)
    print(f"Unique queries after normalization: {len(unique_queries)}")
    for i, query in enumerate(unique_queries, 1):
        print(f"{i}. {query}")
    print()

    docs = retrieve_with_deduplicated_queries(base_retriever, unique_queries)
    print(f"Retrieved {len(docs)} documents:")
    for i, doc in enumerate(docs, 1):
        print(f"{i}. {doc.page_content}")


if __name__ == "__main__":
    run_example()
