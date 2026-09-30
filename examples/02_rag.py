"""RAG example: retrieve and generate with graceful empty handling."""

from typing import Callable, List

from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.llms.base import BaseLLM
from langchain.schema import BaseRetriever, Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS


def format_docs(docs: List[Document]) -> str:
    """Format retrieved documents into a single context string.

    Args:
        docs: List of documents retrieved for a query.

    Returns:
        A newline-separated string combining the content of each document.
        Returns an empty string when ``docs`` is empty.
    """
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(
    retriever: BaseRetriever,
    llm: BaseLLM,
    document_formatter: Callable[[List[Document]], str] = format_docs,
) -> Callable[[str], str]:
    """Build a RAG chain that retrieves documents and generates an answer.

    Args:
        retriever: A retriever with a ``get_relevant_documents`` method.
        llm: A language model instance used to generate the final answer.
        document_formatter: Optional callable that turns retrieved documents
            into a single context string. Defaults to :func:`format_docs`.

    Returns:
        A function that accepts a query string and returns the generated answer.
    """

    def rag_chain(query: str) -> str:
        docs = retriever.get_relevant_documents(query)
        if not docs:
            return "I couldn't find any relevant information to answer your question."

        context = document_formatter(docs)
        prompt = (
            f"Based on the following context, answer the question.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        )
        return llm(prompt)

    return rag_chain


def load_documents(file_path: str) -> List[Document]:
    """Load documents from a file and split them into chunks.

    Args:
        file_path: Path to the text file to load.

    Returns:
        A list of split Document objects ready for indexing.
    """
    loader = TextLoader(file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_documents(documents)


def create_retriever(documents: List[Document]) -> BaseRetriever:
    """Create a vector store retriever from a list of documents.

    Args:
        documents: Documents to embed and index.

    Returns:
        A retriever backed by a FAISS vector store.
    """
    # Create embeddings to convert document chunks into vector representations.
    embeddings = OpenAIEmbeddings()

    # Build a FAISS vector store from the documents and their embeddings.
    # FAISS is a library for efficient similarity search and clustering of dense vectors.
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Return a retriever interface that can be used to query the vector store.
    return vectorstore.as_retriever()


def load_and_index_documents(file_path: str) -> BaseRetriever:
    """Load, split, and index documents into a retriever.

    This is a convenience wrapper around :func:`load_documents` and
    :func:`create_retriever`.
    """
    documents = load_documents(file_path)
    return create_retriever(documents)


def main() -> None:
    # Load, split, and index documents
    documents = load_documents("data.txt")
    retriever = create_retriever(documents)

    # Initialize LLM
    llm = OpenAI(temperature=0)

    # Build the RAG chain
    rag_chain = build_rag_chain(retriever, llm)

    # Query
    query = "What is the capital of France?"
    response = rag_chain(query)
    print(response)


if __name__ == "__main__":
    main()
