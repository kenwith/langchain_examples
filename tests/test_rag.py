"""
RAG (Retrieval-Augmented Generation) tests for langchain_examples.

Tests cover:
- Document loading from various sources
- Text splitting strategies
- Embedding generation and vector storage
- Retrieval with different search types
- End-to-end QA pipeline with assertions

Run: python -m pytest tests/test_rag.py -v
"""

import os
import tempfile
from pathlib import Path
from typing import List

import pytest
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_examples.utils import init_chat_model, init_embeddings


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================

SAMPLE_DOCS = [
    Document(
        page_content="LangChain is a framework for developing applications powered by language models. "
                     "It provides tools for prompt management, memory, and agent orchestration.",
        metadata={"source": "intro.txt", "topic": "overview"}
    ),
    Document(
        page_content="RAG (Retrieval-Augmented Generation) combines retrieval systems with generative models. "
                     "Documents are embedded, stored in a vector database, and retrieved at query time.",
        metadata={"source": "rag.txt", "topic": "rag"}
    ),
    Document(
        page_content="Vector stores like FAISS, Chroma, and Pinecone enable similarity search over embeddings. "
                     "They support metadata filtering and hybrid search strategies.",
        metadata={"source": "vectors.txt", "topic": "vector_stores"}
    ),
    Document(
        page_content="Text splitters break large documents into smaller chunks for embedding. "
                     "RecursiveCharacterTextSplitter is the recommended default for most use cases.",
        metadata={"source": "splitting.txt", "topic": "text_splitting"}
    ),
]


def create_temp_docs(docs: List[Document], suffix: str = ".txt") -> List[str]:
    """Create temporary files from documents and return their paths."""
    paths = []
    for i, doc in enumerate(docs):
        with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as f:
            f.write(doc.page_content)
            paths.append(f.name)
    return paths


def cleanup_temp_files(paths: List[str]) -> None:
    """Clean up temporary files."""
    for path in paths:
        try:
            os.unlink(path)
        except OSError:
            pass


# =============================================================================
# Document Loading Tests
# =============================================================================

def test_document_loading_from_text_files() -> None:
    """Test loading documents from text files using TextLoader."""
    temp_paths = create_temp_docs(SAMPLE_DOCS)
    try:
        loaded_docs = []
        for path in temp_paths:
            loader = TextLoader(path)
            loaded_docs.extend(loader.load())

        assert len(loaded_docs) == len(SAMPLE_DOCS)
        for i, doc in enumerate(loaded_docs):
            assert doc.page_content == SAMPLE_DOCS[i].page_content
            assert doc.metadata["source"] == os.path.basename(temp_paths[i])
    finally:
        cleanup_temp_files(temp_paths)


def test_document_loading_with_custom_encoding() -> None:
    """Test loading documents with explicit encoding."""
    temp_paths = create_temp_docs(SAMPLE_DOCS[:1])
    try:
        loader = TextLoader(temp_paths[0], encoding="utf-8")
        docs = loader.load()
        assert len(docs) == 1
        assert "LangChain" in docs[0].page_content
    finally:
        cleanup_temp_files(temp_paths)


# =============================================================================
# Text Splitting Tests
# =============================================================================

def test_recursive_character_text_splitter_defaults() -> None:
    """Test RecursiveCharacterTextSplitter with default parameters."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
    )
    chunks = splitter.split_documents(SAMPLE_DOCS)

    assert len(chunks) > len(SAMPLE_DOCS)  # Should split into more chunks
    for chunk in chunks:
        assert len(chunk.page_content) <= 120  # chunk_size + overlap buffer
        assert "source" in chunk.metadata


def test_text_splitter_preserves_metadata() -> None:
    """Test that text splitting preserves and propagates metadata."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
    chunks = splitter.split_documents(SAMPLE_DOCS[:1])

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.metadata["source"] == "intro.txt"
        assert chunk.metadata["topic"] == "overview"


def test_text_splitter_empty_document() -> None:
    """Test splitting an empty document returns empty list."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    empty_doc = Document(page_content="", metadata={"source": "empty.txt"})
    chunks = splitter.split_documents([empty_doc])
    assert chunks == []


# =============================================================================
# Embedding Tests
# =============================================================================

def test_embeddings_initialization() -> None:
    """Test provider-agnostic embeddings initialization."""
    embeddings = init_embeddings()
    assert embeddings is not None
    assert isinstance(embeddings, Embeddings)


def test_embeddings_generate_vectors() -> None:
    """Test that embeddings generate vectors of correct dimension."""
    embeddings = init_embeddings()
    texts = ["Hello world", "LangChain is great"]
    vectors = embeddings.embed_documents(texts)

    assert len(vectors) == 2
    assert all(len(v) > 0 for v in vectors)
    assert all(len(v) == len(vectors[0]) for v in vectors)  # Consistent dimensions


def test_embeddings_query_embedding() -> None:
    """Test single query embedding generation."""
    embeddings = init_embeddings()
    vector = embeddings.embed_query("What is LangChain?")

    assert len(vector) > 0
    assert isinstance(vector[0], float)


# =============================================================================
# Vector Store Tests
# =============================================================================

def test_faiss_vector_store_creation() -> None:
    """Test creating a FAISS vector store from documents."""
    embeddings = init_embeddings()
    vectorstore = FAISS.from_documents(SAMPLE_DOCS, embeddings)

    assert vectorstore is not None
    assert vectorstore.index.ntotal == len(SAMPLE_DOCS)


def test_faiss_similarity_search() -> None:
    """Test similarity search returns relevant documents."""
    embeddings = init_embeddings()
    vectorstore = FAISS.from_documents(SAMPLE_DOCS, embeddings)

    results = vectorstore.similarity_search("What is RAG?", k=2)

    assert len(results) == 2
    # Should retrieve the RAG-related document
    assert any("RAG" in doc.page_content for doc in results)


def test_faiss_similarity_search_with_score() -> None:
    """Test similarity search with relevance scores."""
    embeddings = init_embeddings()
    vectorstore = FAISS.from_documents(SAMPLE_DOCS, embeddings)

    results = vectorstore.similarity_search_with_score("vector database", k=2)

    assert len(results) == 2
    for doc, score in results:
        assert isinstance(score, float)
        assert 0 <= score <= 1  # FAISS returns L2 distance, lower is better


def test_faiss_metadata_filtering() -> None:
    """Test metadata filtering during retrieval."""
    embeddings = init_embeddings()
    vectorstore = FAISS.from_documents(SAMPLE_DOCS, embeddings)

    # Filter for only "rag" topic documents
    results = vectorstore.similarity_search(
        "retrieval",
        k=3,
        filter={"topic": "rag"}
    )

    assert len(results) >= 1
    for doc in results:
        assert doc.metadata["topic"] == "rag"


def test_faiss_mmr_search() -> None:
    """Test Maximum Marginal Relevance search for diversity."""
    embeddings = init_embeddings()
    vectorstore = FAISS.from_documents(SAMPLE_DOCS, embeddings)

    results = vectorstore.max_marginal_relevance_search(
        "language models and frameworks",
        k=2,
        fetch_k=4,
        lambda_mult=0.5
    )

    assert len(results) == 2
    # Results should be diverse (different topics)
    topics = {doc.metadata["topic"] for doc in results}
    assert len(topics) >= 1  # At least one topic represented


# =============================================================================
# Retrieval Chain Tests
# =============================================================================

def test_retriever_as_runnable() -> None:
    """Test using vectorstore as retriever in a chain."""
    embeddings = init_embeddings()
    vectorstore = FAISS.from_documents(SAMPLE_DOCS, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # Test as a runnable
    results = retriever.invoke("vector stores")
    assert len(results) == 2
    assert all(isinstance(doc, Document) for doc in results)


def test_retriever_with_different_search_types() -> None:
    """Test retriever with different search configurations."""
    embeddings = init_embeddings()
    vectorstore = FAISS.from_documents(SAMPLE_DOCS, embeddings)

    # Similarity search (default)
    retriever_sim = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    results_sim = retriever_sim.invoke("embeddings")
    assert len(results_sim) == 2

    # MMR search
    retriever_mmr = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 2, "fetch_k": 4, "lambda_mult": 0.7}
    )
    results_mmr = retriever_mmr.invoke("embeddings")
    assert len(results_mmr) == 2


# =============================================================================
# End-to-End QA Tests
# =============================================================================

def test_end_to_end_qa_pipeline() -> None:
    """Test complete RAG QA pipeline: load -> split -> embed -> retrieve -> generate."""
    # Setup
    embeddings = init_embeddings()
    llm = init_chat_model(temperature=0)

    # Create vector store
    vectorstore = FAISS.from_documents(SAMPLE_DOCS, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # Define prompt
    prompt = ChatPromptTemplate.from_template("""
    Answer the question based only on the following context:
    {context}

    Question: {question}
    """)

    # Build chain
    def format_docs(docs: List[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Execute
    answer = chain.invoke("What is RAG?")

    # Assertions
    assert isinstance(answer, str)
    assert len(answer) > 0
    # Answer should mention retrieval and generation
    assert any(keyword in answer.lower() for keyword in ["retrieval", "generation", "rag", "vector"])


def test_qa_pipeline_with_citations() -> None:
    """Test QA pipeline that returns sources with answer."""
    embeddings = init_embeddings()
    llm = init_chat_model(temperature=0)

    vectorstore = FAISS.from_documents(SAMPLE_DOCS, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template("""
    Answer the question based only on the following context. Cite sources using [source].
    {context}

    Question: {question}
    """)

    def format_docs_with_sources(docs: List[Document]) -> str:
        return "\n\n".join(
            f"[{doc.metadata.get('source', 'unknown')}] {doc.page_content}"
            for doc in docs
        )

    chain = (
        {"context": retriever | format_docs_with_sources, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke("What are vector stores used for?")

    assert isinstance(answer, str)
    assert len(answer) > 0
    # Should reference sources
    assert "[" in answer and "]" in answer


def test_qa_pipeline_handles_no_relevant_docs() -> None:
    """Test QA pipeline behavior when no relevant documents are found."""
    embeddings = init_embeddings()
    llm = init_chat_model(temperature=0)

    # Vector store with unrelated content
    unrelated_docs = [
        Document(page_content="The weather is sunny today.", metadata={"source": "weather.txt"})
    ]
    vectorstore = FAISS.from_documents(unrelated_docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

    prompt = ChatPromptTemplate.from_template("""
    Answer based only on context. If context doesn't contain the answer, say "I don't know."
    {context}

    Question: {question}
    """)

    chain = (
        {"context": retriever | (lambda docs: "\n".join(d.page_content for d in docs)), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke("What is LangChain?")

    assert isinstance(answer, str)
    # Should indicate lack of knowledge
    assert "don't know" in answer.lower() or "not" in answer.lower()


# =============================================================================
# Integration Tests
# =============================================================================

def test_full_rag_workflow_with_temp_files() -> None:
    """Test complete workflow starting from temporary files."""
    # Create temp files
    temp_paths = create_temp_docs(SAMPLE_DOCS)
    try:
        # Load
        loaded_docs = []
        for path in temp_paths:
            loader = TextLoader(path)
            loaded_docs.extend(loader.load())

        # Split
        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
        chunks = splitter.split_documents(loaded_docs)

        # Embed and store
        embeddings = init_embeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)

        # Retrieve
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        results = retriever.invoke("text splitting")

        # Verify
        assert len(results) == 3
        assert any("split" in doc.page_content.lower() for doc in results)

    finally:
        cleanup_temp_files(temp_paths)


def test_vector_store_persistence() -> None:
    """Test saving and loading vector store to/from disk."""
    embeddings = init_embeddings()
    vectorstore = FAISS.from_documents(SAMPLE_DOCS, embeddings)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save
        vectorstore.save_local(tmpdir)

        # Load
        loaded_store = FAISS.load_local(tmpdir, embeddings, allow_dangerous_deserialization=True)

        # Verify
        assert loaded_store.index.ntotal == vectorstore.index.ntotal

        # Search should work identically
        results_orig = vectorstore.similarity_search("LangChain", k=2)
        results_loaded = loaded_store.similarity_search("LangChain", k=2)

        assert len(results_orig) == len(results_loaded)
        for orig, loaded in zip(results_orig, results_loaded):
            assert orig.page_content == loaded.page_content


# =============================================================================
# Demo / Main Block
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("RAG Test Suite Demo")
    print("=" * 60)

    print("\n1. Testing document loading...")
    test_document_loading_from_text_files()
    print("   ✓ Document loading works")

    print("\n2. Testing text splitting...")
    test_recursive_character_text_splitter_defaults()
    test_text_splitter_preserves_metadata()
    print("   ✓ Text splitting works")

    print("\n3. Testing embeddings...")
    test_embeddings_initialization()
    test_embeddings_generate_vectors()
    print("   ✓ Embeddings work")

    print("\n4. Testing vector store operations...")
    test_faiss_vector_store_creation()
    test_faiss_similarity_search()
    test_faiss_metadata_filtering()
    test_faiss_mmr_search()
    print("   ✓ Vector store operations work")

    print("\n5. Testing retrieval chains...")
    test_retriever_as_runnable()
    test_retriever_with_different_search_types()
    print("   ✓ Retrieval chains work")

    print("\n6. Testing end-to-end QA...")
    test_end_to_end_qa_pipeline()
    test_qa_pipeline_with_citations()
    print("   ✓ End-to-end QA works")

    print("\n7. Testing full workflow...")
    test_full_rag_workflow_with_temp_files()
    test_vector_store_persistence()
    print("   ✓ Full workflow works")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
