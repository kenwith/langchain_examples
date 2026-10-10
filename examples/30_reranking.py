"""Example: Retrieval with reranking.

This example demonstrates how to add a reranking step to improve retrieval quality.
It uses a cross-encoder model to rerank documents retrieved by a vector store.
The example includes helper functions `build_reranker` and `rerank_documents` to make
the reranking step easy to reuse, plus `format_documents` and `print_results` for
readable output. It also evaluates retrieval quality before and after reranking.
"""

from typing import Callable, List, Sequence, Set

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


def build_reranker(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k: int = 3,
) -> Callable[..., List[Document]]:
    """Build a reranker function.

    The returned function reranks documents by relevance to the query using a
    cross-encoder. If the cross-encoder model cannot be loaded (for example, if
    sentence-transformers is not installed or the model is unavailable), it falls
    back to returning the first `top_k` documents unchanged. When the model is
    available, the reranked documents include a `rerank_score` metadata field.

    Args:
        model_name: Name of the cross-encoder model to use.
        top_k: Default number of top documents to return.

    Returns:
        A function with signature (query, documents, top_k=...) -> List[Document].
    """
    try:
        from sentence_transformers import CrossEncoder

        encoder = CrossEncoder(model_name)
    except (ImportError, OSError, RuntimeError, ValueError):
        encoder = None

    def rerank_documents(
        query: str,
        documents: Sequence[Document],
        top_k: int = top_k,
    ) -> List[Document]:
        """Rerank documents by relevance to the query.

        Args:
            query: The query string.
            documents: The documents to rerank.
            top_k: The number of top documents to return.

        Returns:
            A list of documents sorted by relevance (most relevant first).
        """
        if encoder is None:
            return list(documents[:top_k])

        pairs = [(query, doc.page_content) for doc in documents]
        scores = encoder.predict(pairs)

        # Sort documents by score in descending order.
        scored = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

        # Create copies with the rerank score attached to metadata.
        reranked = []
        for doc, score in scored[:top_k]:
            metadata = dict(doc.metadata)
            metadata["rerank_score"] = score
            reranked.append(
                Document(page_content=doc.page_content, metadata=metadata)
            )
        return reranked

    return rerank_documents


_DEFAULT_RERANKER = build_reranker()


def rerank_documents(
    query: str,
    documents: Sequence[Document],
    top_k: int = 3,
) -> List[Document]:
    """Rerank documents by relevance to the query using a cross-encoder.

    This is a convenience wrapper around `build_reranker`.

    Args:
        query: The query string.
        documents: The documents to rerank.
        top_k: The number of top documents to return.

    Returns:
        A list of documents sorted by relevance (most relevant first).
    """
    return _DEFAULT_RERANKER(query, documents, top_k)


def format_documents(documents: Sequence[Document]) -> str:
    """Format documents for readable output."""
    lines = []
    for i, doc in enumerate(documents, start=1):
        lines.append(f"Document {i}:")
        lines.append(f"  Source: {doc.metadata.get('source', 'unknown')}")
        score = doc.metadata.get("rerank_score")
        if score is not None:
            lines.append(f"  Rerank Score: {score:.4f}")
        lines.append(f"  Content: {doc.page_content[:200]}...")
        lines.append("")
    return "\n".join(lines)


def print_results(title: str, documents: Sequence[Document]) -> None:
    """Print a title and formatted documents.

    Args:
        title: The title to print before the documents.
        documents: The documents to format and print.
    """
    print(title)
    print(format_documents(documents))
    print()


def precision_at_k(
    documents: Sequence[Document],
    relevant_sources: Set[str],
    k: int,
) -> float:
    """Compute precision at k for a ranked list of documents.

    Args:
        documents: The ranked documents.
        relevant_sources: A set of source metadata values considered relevant.
        k: The number of top documents to consider.

    Returns:
        The fraction of documents in the top k that are relevant.
    """
    if k <= 0:
        return 0.0
    top_k = documents[:k]
    if not top_k:
        return 0.0
    relevant = sum(
        1 for doc in top_k if doc.metadata.get("source") in relevant_sources
    )
    return relevant / k


def main() -> None:
    # Sample documents. The fifth document is a distractor: it contains the same
    # keywords as the query but is not about actual foxes jumping over dogs.
    documents = [
        Document(
            page_content="Foxes jump over dogs to practice their agility.",
            metadata={"source": "example1.txt"},
        ),
        Document(
            page_content="A fox may leap over a dog during play.",
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
        Document(
            page_content=(
                "The phrase 'the quick brown fox jumps over the lazy dog' "
                "is a common typing exercise."
            ),
            metadata={"source": "example5.txt"},
        ),
    ]

    # Split documents into smaller chunks (optional)
    splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
    chunks = splitter.split_documents(documents)

    # Create a vector store (using OpenAI embeddings)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    query = "Why do foxes jump over dogs?"
    relevant_sources = {"example1.txt", "example2.txt"}

    # Retrieve initial documents
    initial_docs = retriever.invoke(query)

    print_results("=== Initial Retrieval ===", initial_docs)
    print(f"Precision@2 (initial): {precision_at_k(initial_docs, relevant_sources, 2):.2f}")
    print()

    # Build a reranker and rerank the retrieved documents
    rerank = build_reranker()
    reranked_docs = rerank(query, initial_docs, top_k=2)

    print_results("=== After Reranking ===", reranked_docs)
    print(f"Precision@2 (reranked): {precision_at_k(reranked_docs, relevant_sources, 2):.2f}")
    print()


if __name__ == "__main__":
    main()
