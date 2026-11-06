"""Retrieval-augmented generation (RAG) example using LangChain.

This script builds a Chroma vector store from text files and answers questions
using a chat model with retrieval context. It can be run in two modes:

1. Create/update the vector store:
   python examples/02_rag.py --create [directory_path]

   If directory_path is omitted, the script uses built-in sample texts.

2. Query an existing vector store:
   python examples/02_rag.py

Setup:
- Install the required packages:
  pip install langchain langchain-community langchain-openai chromadb
  Additional provider packages may be required for other chat models.
- The OPENAI_API_KEY environment variable must be set to a valid OpenAI API key for embeddings.
- Optional: Set CHAT_MODEL and CHAT_MODEL_PROVIDER to use a different chat model. Provider-specific API keys may also need to be set.
- Optional: Place text files in a directory to use as the knowledge base.

The script includes a reusable `format_docs` helper that joins the page content
of retrieved documents into a single string, which is used in the RAG chain to
provide context for the chat model.
"""
import argparse
import os
import sys
from collections.abc import Iterable

from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Default persist directory for the Chroma vector store.
PERSIST_DIRECTORY = "./chroma_db"

# Sample texts used when no directory is provided.
SAMPLE_TEXTS = [
    "LangChain is a framework for developing applications powered by language models.",
    "Retrieval-augmented generation (RAG) combines a retrieval system with a generative model.",
    "Chroma is a vector database for storing and retrieving embeddings.",
    "The OpenAI API provides access to powerful language models for natural language processing.",
]


def _sample_documents() -> list[Document]:
    """Create a list of sample documents from the built-in texts.

    Returns:
        A list of Document objects created from SAMPLE_TEXTS.
    """
    return [Document(page_content=text) for text in SAMPLE_TEXTS]


def load_documents_from_directory(directory_path: str) -> list[Document]:
    """Load all text documents from a directory.

    Args:
        directory_path: Path to a directory containing text files.

    Returns:
        A list of Document objects loaded from all .txt files in the directory.
    """
    loader = DirectoryLoader(directory_path, glob="**/*.txt", loader_cls=TextLoader)
    return loader.load()


def load_documents(directory_path: str | None) -> list[Document]:
    """Load documents from a directory or return built-in sample texts.

    Args:
        directory_path: Optional path to a directory containing text files.
            If None, a list of sample documents is returned.

    Returns:
        A list of Document objects loaded from the directory or the built-in samples.
    """
    if directory_path:
        return load_documents_from_directory(directory_path)
    return _sample_documents()


def load_documents_from_args(args: argparse.Namespace) -> list[Document]:
    """Load documents based on parsed command-line arguments.

    Args:
        args: Parsed arguments from argparse.

    Returns:
        A list of Document objects loaded from the directory specified by
        ``args.create``, or the built-in sample texts if no directory is given.
    """
    directory_path = args.create if args.create != "" else None
    return load_documents(directory_path)


def create_vector_store(
    documents: list[Document], persist_directory: str = PERSIST_DIRECTORY
) -> Chroma:
    """Create and persist a Chroma vector store from the given documents.

    Args:
        documents: A list of Document objects to index.
        persist_directory: Directory where the Chroma vector store is persisted.

    Returns:
        A Chroma vector store containing the embedded documents.
    """
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    return vectorstore


def load_vector_store(persist_directory: str = PERSIST_DIRECTORY) -> Chroma:
    """Load an existing Chroma vector store from disk.

    Args:
        persist_directory: Directory where the Chroma vector store is persisted.

    Returns:
        The Chroma vector store loaded from the specified directory.
    """
    embeddings = OpenAIEmbeddings()
    return Chroma(
        persist_directory=persist_directory, embedding_function=embeddings
    )


def format_docs(docs: Iterable[Document], separator: str = "\n\n") -> str:
    """Join the content of retrieved documents into a single string.

    Args:
        docs: An iterable of Document objects retrieved from the vector store.
        separator: The string used to separate document contents.

    Returns:
        A single string containing the page content of each document,
        separated by `separator`.
    """
    return separator.join(doc.page_content for doc in docs)


def main() -> None:
    """Run the RAG example in create or query mode."""
    parser = argparse.ArgumentParser(description="RAG example using LangChain.")
    parser.add_argument(
        "--create",
        nargs="?",
        const="",
        metavar="directory_path",
        help="Create or update the vector store from text files in the given directory. "
        "If no directory is provided, built-in sample texts are used.",
    )
    args = parser.parse_args()

    if args.create is not None:
        # Create mode: load documents and build the vector store.
        documents = load_documents_from_args(args)
        print(f"Loaded {len(documents)} documents.")
        create_vector_store(documents)
        print(f"Vector store created/updated at {PERSIST_DIRECTORY}.")
        return

    # Query mode: load the existing vector store and answer questions.
    if not os.path.exists(PERSIST_DIRECTORY):
        print(
            "Vector store not found. Run the script with --create to build it first.",
            file=sys.stderr,
        )
        sys.exit(1)

    vectorstore = load_vector_store()
    retriever = vectorstore.as_retriever()

    # Use init_chat_model to allow provider selection via environment variables.
    chat_model = init_chat_model(
        model=os.getenv("CHAT_MODEL", "gpt-3.5-turbo"),
        model_provider=os.getenv("CHAT_MODEL_PROVIDER") or None,
        temperature=0,
    )

    prompt = PromptTemplate.from_template(
        "Answer the question based only on the following context:\n\n"
        "{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    )
    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | chat_model
        | StrOutputParser()
    )

    print("RAG query mode. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            question = input("\nQuestion: ").strip()
        except EOFError:
            break
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        answer = qa_chain.invoke(question)
        print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
