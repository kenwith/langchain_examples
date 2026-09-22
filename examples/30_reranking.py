"""Rerank retrieved documents using a cross-encoder to improve answer quality.

This example demonstrates how to use LangChain's ContextualCompressionRetriever
with a CrossEncoderReranker to reorder documents retrieved by a vector store.
The reranked documents are then passed to a chat model to generate a grounded answer.

Requirements:
- Set OPENAI_API_KEY for the chat model (or configure another provider).
- The embeddings and cross-encoder models are downloaded from HuggingFace Hub.
"""

import os

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------------------------------------------
# 1. Load and split documents
# ---------------------------------------------------------------
def load_documents() -> list[Document]:
    """Return a small set of sample documents to index."""
    text = """
LangChain is a framework for developing applications powered by language models.
It provides standard interfaces for chains, agents, and retrieval-augmented generation (RAG).
RAG combines a retrieval step with a generation step to produce answers grounded in external knowledge.
The retrieval step typically uses a vector store to find relevant documents based on embedding similarity.
However, embedding similarity can sometimes retrieve documents that are not truly relevant to the query.
Cross-encoders can rerank the retrieved documents by jointly encoding the query and each document.
This reranking step often improves the quality of the final answer.
LangChain integrates with many vector stores and embedding models.
It also supports various chat models through a unified interface.
The framework is open-source and has a large community.
"""
    # Add an unrelated paragraph to demonstrate reranking.
    unrelated = """
The weather today is sunny with a chance of rain in the afternoon.
Remember to bring an umbrella if you go outside.
"""
    return [Document(page_content=text), Document(page_content=unrelated)]


def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into smaller chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    return splitter.split_documents(documents)


# ---------------------------------------------------------------
# 2. Build a base retriever
# ---------------------------------------------------------------
def create_retriever(documents: list[Document]):
    """Create an in-memory vector store retriever."""
    embeddings = init_embeddings(
        os.getenv("EMBEDDINGS_MODEL", "huggingface:sentence-transformers/all-MiniLM-L6-v2")
    )
    vectorstore = InMemoryVectorStore.from_documents(documents, embedding=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})


# ---------------------------------------------------------------
# 3. Create a cross-encoder reranker
# ---------------------------------------------------------------
def create_reranker() -> CrossEncoderReranker:
    """Create a cross-encoder reranker using a small model."""
    cross_encoder = HuggingFaceCrossEncoder(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    return CrossEncoderReranker(model=cross_encoder, top_n=3)


def create_compression_retriever(base_retriever) -> ContextualCompressionRetriever:
    """Wrap the base retriever with the reranker."""
    reranker = create_reranker()
    return ContextualCompressionRetriever(
        base_compressor=reranker, base_retriever=base_retriever
    )


# ---------------------------------------------------------------
# 4. Run the example
# ---------------------------------------------------------------
def main() -> None:
    """Run the reranking example end-to-end."""
    print("Loading and splitting documents...")
    docs = load_documents()
    chunks = split_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    print("Building base retriever...")
    base_retriever = create_retriever(chunks)

    print("Creating compression retriever with cross-encoder reranker...")
    compression_retriever = create_compression_retriever(base_retriever)

    query = "What is RAG and why is reranking useful?"
    print(f"\nQuery: {query}\n")

    print("Retrieved documents (before reranking):")
    initial_docs = base_retriever.invoke(query)
    for i, doc in enumerate(initial_docs, 1):
        print(f"{i}. {doc.page_content[:80]}...")

    print("\nReranked documents:")
    reranked_docs = compression_retriever.invoke(query)
    for i, doc in enumerate(reranked_docs, 1):
        print(f"{i}. {doc.page_content[:80]}...")

    print("\nGenerating answer with chat model...")
    chat_model = init_chat_model(
        os.getenv("CHAT_MODEL", "gpt-4o-mini"),
        model_provider=os.getenv("CHAT_MODEL_PROVIDER", "openai"),
    )
    context = "\n\n".join(doc.page_content for doc in reranked_docs)
    messages = [
        (
            "system",
            "You are a helpful assistant. Answer the question using only the provided context.",
        ),
        ("human", f"Context:\n{context}\n\nQuestion: {query}"),
    ]
    answer = chat_model.invoke(messages)
    print(f"\nAnswer: {answer.content}")


if __name__ == "__main__":
    main()
