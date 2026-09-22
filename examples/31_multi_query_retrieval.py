"""
Example 31: Multi-Query Retrieval
================================

| Example | Description |
|---------|-------------|
| 31 | Multi-Query Retrieval |

This example demonstrates how to use multi-query retrieval to improve recall
by generating multiple queries from an original question.
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


def run_example() -> None:
    """Run the multi-query retrieval example."""
    # Provider-agnostic chat model initialization
    llm = init_chat_model(os.getenv("MODEL", "gpt-4o"), temperature=0)

    vectorstore = create_vectorstore()

    # Build a MultiQueryRetriever from the base retriever and the LLM
    retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        llm=llm,
    )

    question = "What is multi-query retrieval and how does it improve recall?"
    print(f"Question: {question}\n")

    docs = retriever.invoke(question)
    print(f"Retrieved {len(docs)} documents:")
    for i, doc in enumerate(docs, 1):
        print(f"{i}. {doc.page_content}")


if __name__ == "__main__":
    run_example()
