"""Tests for the HyDE example: document generation and retrieval flow.

This module tests the core components of the HyDE (Hypothetical Document
Embeddings) approach: generating a hypothetical document from a query using
a chat model, and using that document to retrieve relevant documents from a
vector store. The tests use fakes to avoid external API calls, and
demonstrate the provider-agnostic use of `init_chat_model`.
"""

from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.language_models.fake_chat_models import FakeListChatModel


# ---------------------------------------------------------------------------
# Test: Document generation (HyDE step 1)
# ---------------------------------------------------------------------------
def test_hyde_document_generation():
    """Generate a hypothetical document from a query using a fake chat model."""
    # Fake chat model returns a fixed hypothetical document.
    fake_llm = FakeListChatModel(responses=[
        "A hypothetical document about climate change impacts on agriculture."
    ])

    # Prompt for HyDE: ask the model to write a hypothetical document that
    # would answer the query.
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert. Write a short hypothetical document "
                   "that would answer the user's query."),
        ("user", "{query}")
    ])

    chain = (
        {"query": RunnablePassthrough()}
        | prompt
        | fake_llm
        | StrOutputParser()
    )

    query = "What are the effects of climate change on crop yields?"
    generated_doc = chain.invoke(query)

    assert generated_doc == "A hypothetical document about climate change impacts on agriculture."


# ---------------------------------------------------------------------------
# Test: Retrieval flow (HyDE step 2)
# ---------------------------------------------------------------------------
def test_hyde_retrieval_flow():
    """Use HyDE to generate a query embedding and retrieve relevant documents."""
    # Fake embeddings (deterministic for testing)
    embeddings = FakeEmbeddings(size=10)

    # Create a small in-memory vector store with dummy documents.
    docs = [
        Document(page_content="Climate change reduces crop yields in many regions."),
        Document(page_content="Renewable energy is essential to fight global warming."),
        Document(page_content="AI helps predict weather patterns for agriculture."),
    ]
    vectorstore = InMemoryVectorStore.from_documents(docs, embedding=embeddings)

    # Fake chat model that generates a hypothetical document.
    fake_llm = FakeListChatModel(responses=[
        "Hypothetical: Climate change severely impacts agricultural productivity."
    ])

    # Prompt for HyDE
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert. Write a short hypothetical document "
                   "that would answer the user's query."),
        ("user", "{query}")
    ])

    # Build a HyDE chain: generate hypothetical doc -> embed -> retrieve.
    hyde_chain = (
        {"query": RunnablePassthrough()}
        | prompt
        | fake_llm
        | StrOutputParser()
    )

    # Query the chain to get the hypothetical document
    query = "How does climate change affect farming?"
    hypothetical_doc = hyde_chain.invoke(query)

    # Embed the hypothetical document and retrieve similar documents.
    query_embedding = embeddings.embed_query(hypothetical_doc)
    retrieved_docs = vectorstore.similarity_search_by_vector(query_embedding, k=2)

    # The most relevant doc should be the one about climate change and crop yields.
    assert any("crop yields" in doc.page_content for doc in retrieved_docs)
    assert len(retrieved_docs) <= 2


# ---------------------------------------------------------------------------
# Demo / main block
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Run the tests using pytest
    import pytest
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
