"""
Example 58: Ollama Reranking

| Field       | Value                                                                        |
|-------------|------------------------------------------------------------------------------|
| Example     | 58                                                                           |
| Name        | Ollama Reranking                                                             |
| Description | Combines Ollama embeddings with a local cross-encoder for reranking.         |
| Dependencies| langchain, langchain-ollama, langchain-community, sentence-transformers, chromadb |
| Run         | python examples/58_ollama_reranking.py                                       |
"""

import os
from typing import List, Sequence

from langchain.chat_models import init_chat_model
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder


def load_sample_documents() -> List[Document]:
    """Create a small set of sample documents for the demo."""
    return [
        Document(page_content="Ollama is a tool for running large language models locally."),
        Document(page_content="LangChain is a framework for building applications with language models."),
        Document(page_content="Reranking is a technique to improve retrieval quality by reordering documents."),
        Document(page_content="Cross-encoders are models that score a query-document pair directly."),
        Document(page_content="Embeddings represent text as vectors for semantic search."),
    ]


def create_vectorstore(documents: List[Document], embeddings: OllamaEmbeddings) -> Chroma:
    """Split documents and create an in-memory Chroma vector store."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    splits = splitter.split_documents(documents)
    return Chroma.from_documents(splits, embeddings)


def retrieve_documents(query: str, retriever: BaseRetriever, k: int = 5) -> List[Document]:
    """Retrieve the top k documents using a vector store retriever."""
    return retriever.invoke(query)


def rerank_documents(
    query: str,
    documents: Sequence[Document],
    cross_encoder: CrossEncoder,
) -> List[Document]:
    """Rerank documents using a local cross-encoder."""
    pairs = [(query, doc.page_content) for doc in documents]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked]


def generate_answer(query: str, documents: Sequence[Document], chat_model) -> str:
    """Generate an answer from the reranked documents."""
    context = "\n\n".join(doc.page_content for doc in documents)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Answer the question using only the provided context."),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ]
    )
    chain = prompt | chat_model | StrOutputParser()
    return chain.invoke({"context": context, "question": query})


def main() -> None:
    """Run the Ollama reranking example."""
    # Initialize embeddings and cross-encoder.
    embeddings = OllamaEmbeddings(
        model=os.getenv("OLLAMA_EMBEDDINGS_MODEL", "nomic-embed-text")
    )
    cross_encoder = CrossEncoder(
        os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    )

    # Create vector store and retriever.
    documents = load_sample_documents()
    vectorstore = create_vectorstore(documents, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Retrieve and rerank.
    query = "How does reranking improve retrieval?"
    retrieved = retrieve_documents(query, retriever)
    reranked = rerank_documents(query, retrieved, cross_encoder)

    # Generate answer using a provider-agnostic chat model.
    chat_model = init_chat_model(
        model=os.getenv("OLLAMA_MODEL", "llama3.2"),
        model_provider="ollama",
        temperature=0,
    )
    answer = generate_answer(query, reranked, chat_model)

    print(f"Query: {query}")
    print(f"Retrieved {len(retrieved)} documents, reranked to {len(reranked)} documents.")
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
