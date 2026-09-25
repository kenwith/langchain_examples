"""Example: Retrieval with reranking.

This example demonstrates how to add a reranking step to improve retrieval quality.
It includes a helper function `rerank_documents` and improved output formatting.
"""

from typing import List, Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Optional: use a cross-encoder for reranking. If not available, fallback.
try:
    from sentence_transformers import CrossEncoder

    _CROSS_ENCODER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
except ImportError:
    _CROSS_ENCODER = None


def rerank_documents(
    query: str,
    documents: Sequence[Document],
    top_k: int = 3,
) -> List[Document]:
    """Rerank documents by relevance to the query using a cross-encoder.

    Args:
        query: The query string.
        documents: The documents to rerank.
        top_k: The number of top documents to return.

    Returns:
        A list of documents sorted by relevance (most relevant first).
    """
    if _CROSS_ENCODER is None:
        # Fallback: return the first top_k documents unchanged.
        return list(documents[:top_k])

    pairs = [(query, doc.page_content) for doc in documents]
    scores = _CROSS_ENCODER.predict(pairs)

    # Sort documents by score in descending order.
    scored = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored[:top_k]]


def format_documents(documents: Sequence[Document]) -> str:
    """Format documents for readable output."""
    lines = []
    for i, doc in enumerate(documents, start=1):
        lines.append(f"Document {i}:")
        lines.append(f"  Source: {doc.metadata.get('source', 'unknown')}")
        lines.append(f"  Content: {doc.page_content[:200]}...")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    # Sample documents
    documents = [
        Document(
            page_content="The quick brown fox jumps over the lazy dog.",
            metadata={"source": "example1.txt"},
        ),
        Document(
            page_content="A fast brown fox leaps over a sleepy canine.",
            metadata={"source": "example2.txt"},
        ),
        Document(
            page_content="The weather today is sunny and warm.",
            metadata={"source": "example3.txt"},
        ),
        Document(
            page_content="Dogs are loyal companions and love to play fetch.",
            metadata={"source": "example4.txt"},
        ),
    ]

    # Split documents into smaller chunks (optional)
    splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
    chunks = splitter.split_documents(documents)

    # Create a vector store (using OpenAI embeddings)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    query = "Tell me about foxes jumping over dogs"

    # Retrieve initial documents
    initial_docs = retriever.invoke(query)

    print("=== Initial Retrieval ===")
    print(format_documents(initial_docs))
    print()

    # Rerank the retrieved documents
    reranked_docs = rerank_documents(query, initial_docs, top_k=2)

    print("=== After Reranking ===")
    print(format_documents(reranked_docs))


if __name__ == "__main__":
    main()
