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
- The OPENAI_API_KEY environment variable must be set to a valid OpenAI API key for embeddings.
- Optional: Set CHAT_MODEL and CHAT_MODEL_PROVIDER to use a different chat model. Provider-specific API keys may also need to be set.
- Optional: Place text files in a directory to use as the knowledge base.
"""
import argparse
import os
import sys

from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI

# Default persist directory for the Chroma vector store.
PERSIST_DIRECTORY = "./chroma_db"

# Sample texts used when no directory is provided.
SAMPLE_TEXTS = [
    "LangChain is a framework for developing applications powered by language models.",
    "Retrieval-augmented generation (RAG) combines a retrieval system with a generative model.",
    "Chroma is a vector database for storing and retrieving embeddings.",
    "The OpenAI API provides access to powerful language models for natural language processing.",
]


def load_documents(directory_path: str | None) -> list[Document]:
    """Load documents from a directory or return built-in sample texts.

    Args:
        directory_path: Optional path to a directory containing text files.
            If None, a list of sample documents is returned.

    Returns:
        A list of Document objects loaded from the directory or the built-in samples.
    """
    if directory_path:
        loader = DirectoryLoader(directory_path, glob="**/*.txt", loader_cls=TextLoader)
        return loader.load()
    return [Document(page_content=text) for text in SAMPLE_TEXTS]


def build_retriever(
    documents: list[Document], persist_directory: str = PERSIST_DIRECTORY
) -> VectorStoreRetriever:
    """Build and return a retriever from the given documents.

    Creates a Chroma vector store from the documents, persists it to the
    specified directory, and returns a retriever for querying.

    Args:
        documents: A list of Document objects to index.
        persist_directory: Directory where the Chroma vector store is persisted.

    Returns:
        A retriever object that can be used to retrieve relevant documents.
    """
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    return vectorstore.as_retriever()


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
        directory_path = args.create if args.create != "" else None
        documents = load_documents(directory_path)
        print(f"Loaded {len(documents)} documents.")
        build_retriever(documents)
        print(f"Vector store created/updated at {PERSIST_DIRECTORY}.")
        return

    # Query mode: load the existing vector store and answer questions.
    if not os.path.exists(PERSIST_DIRECTORY):
        print(
            "Vector store not found. Run the script with --create to build it first.",
            file=sys.stderr,
        )
        sys.exit(1)

    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever()

    chat_model = ChatOpenAI(
        model=os.getenv("CHAT_MODEL", "gpt-3.5-turbo"),
        temperature=0,
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=chat_model, chain_type="stuff", retriever=retriever
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
        print(f"Answer: {answer['result']}")


if __name__ == "__main__":
    main()
