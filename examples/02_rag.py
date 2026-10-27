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
import os
import sys

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chat_models import init_chat_model
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

DATA_FILE = "data.txt"
PERSIST_DIR = "db"
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-3.5-turbo")
CHAT_MODEL_PROVIDER = os.getenv("CHAT_MODEL_PROVIDER", "openai")

SAMPLE_TEXTS = [
    "The president delivered a speech to the nation, emphasizing the importance of unity and progress.",
    "In the speech, the president praised the nomination of Ketanji Brown Jackson to the Supreme Court, describing her as a highly qualified and respected jurist.",
    "The president also discussed economic recovery and the need for bipartisan cooperation in Congress.",
    "Ketanji Brown Jackson's confirmation hearings were held in the Senate, where she answered questions about her judicial philosophy and record.",
]


def load_documents_from_directory(directory_path):
    """Load all text documents from a directory.

    Args:
        directory_path: Path to a directory containing .txt files.

    Returns:
        A list of loaded documents, or an empty list if the directory
        does not exist or contains no text files.
    """
    try:
        loader = DirectoryLoader(
            directory_path,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        documents = loader.load()
    except FileNotFoundError:
        print(
            f"Error: Directory '{directory_path}' not found.",
            file=sys.stderr,
        )
        return []

    if not documents:
        print(
            f"No text files found in directory '{directory_path}'.",
            file=sys.stderr,
        )
        return []

    return documents


def load_sample_documents():
    """Create sample documents from built-in texts.

    Returns:
        A list of Document objects containing the sample texts.
    """
    return [
        Document(page_content=text, metadata={"source": "sample"})
        for text in SAMPLE_TEXTS
    ]


def get_source_documents(directory_path=None):
    """Load raw documents from a directory or fall back to sample texts.

    This helper centralizes the source selection logic so callers can obtain
    unfragmented documents without applying text splitting. If a directory is
    provided but cannot be loaded or contains no text files, the built-in
    sample texts are used as a fallback.

    Args:
        directory_path: Optional path to a directory containing .txt files.

    Returns:
        A list of Document objects, or an empty list if no source is available.
    """
    if directory_path:
        documents = load_documents_from_directory(directory_path)
        if not documents:
            print(
                "Falling back to sample texts.",
                file=sys.stderr,
            )
            return load_sample_documents()
        return documents

    return load_sample_documents()


def load_documents(directory_path=None, chunk_size=1000, chunk_overlap=0):
    """Load and split documents from a directory or built-in sample texts.

    Uses `get_source_documents` to select the raw document source, then splits
    the documents into chunks for indexing or retrieval.

    Args:
        directory_path: Optional path to a directory containing text files.
        chunk_size: Maximum size of each text chunk.
        chunk_overlap: Number of characters to overlap between chunks.

    Returns:
        A list of document chunks, or an empty list if no documents are found.
    """
    documents = get_source_documents(directory_path)

    if not documents:
        print("No documents found to load.", file=sys.stderr)
        return []

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)


def create_vector_store(directory_path=None, persist_dir=PERSIST_DIR):
    """Create and persist the vector store from documents.

    Args:
        directory_path: Optional path to a directory containing text files.
            If not provided, sample texts are used.
        persist_dir: Directory where the vector store should be persisted.

    Returns:
        The created Chroma vector store, or None if no documents are available.
    """
    docs = load_documents(directory_path=directory_path)
    if not docs:
        print("No documents found to index. Exiting.", file=sys.stderr)
        return None

    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        docs, embeddings, persist_directory=persist_dir
    )
    vectorstore.persist()
    print(f"Vector store created at {persist_dir}")
    return vectorstore


def format_docs(docs):
    """Format a list of documents into a single string for use as context.

    Each document is prefixed with its source name (from metadata) so the
    model can trace information back to the original file. If no documents
    are provided, a fallback message is returned so the LLM knows there is
    no relevant context to use.

    Args:
        docs: A list of Document objects.

    Returns:
        A string containing the formatted context.
    """
    if not docs:
        return "No relevant documents found in the knowledge base."

    formatted = []
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"Source {idx} ({source}):\n{doc.page_content}")
    return "\n\n".join(formatted)


def build_chain(retriever, llm):
    """Build a RAG chain for retrieval-augmented generation.

    The chain retrieves relevant documents for a user query, formats them as
    context via `format_docs`, and passes the context and question to the
    model using a prompt template.

    Args:
        retriever: A retriever from the vector store.
        llm: A chat model instance.

    Returns:
        A Runnable representing the complete RAG chain.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer questions based on the provided context. "
                "When you use information from a source, cite it by name (e.g., Source 1 (sample)). "
                "At the end of your answer, include a 'Sources:' section that lists every source you used.",
            ),
            ("human", "Context: {context}\n\nQuestion: {input}"),
        ]
    )

    # Retrieval step: fetch relevant documents from the vector store
    # based on the user's query, then format them into a context string
    # using format_docs. If no documents are retrieved, format_docs
    # returns a fallback message.
    # Generation step: pass the context and the original question to the
    # LLM via the prompt template, and generate the final answer.
    return (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["input"]))
        )
        | prompt
        | llm
    )


def main():
    """Run the RAG pipeline."""
    try:
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma(
            persist_directory=PERSIST_DIR, embedding_function=embeddings
        )
    except Exception as e:
        print(
            f"Error: Could not load vector store from '{PERSIST_DIR}'.\n"
            f"Details: {e}\n"
            f"Run `python {os.path.basename(__file__)} --create` to create it.",
            file=sys.stderr,
        )
        return

    if not vectorstore.get()["ids"]:
        print(
            f"Error: The vector store at '{PERSIST_DIR}' is empty.\n"
            f"Run `python {os.path.basename(__file__)} --create` to create it.",
            file=sys.stderr,
        )
        return

    retriever = vectorstore.as_retriever()

    llm = init_chat_model(model=CHAT_MODEL, model_provider=CHAT_MODEL_PROVIDER)

    rag_chain = build_chain(retriever, llm)

    query = "What did the president say about Ketanji Brown Jackson?"
    response = rag_chain.invoke({"input": query})
    answer = response.content

    # Append source metadata to the generated answer so the output remains
    # transparent even if the model does not include citations on its own.
    docs = retriever.invoke(query)
    if docs:
        sources = []
        seen = set()
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            if source not in seen:
                seen.add(source)
                sources.append(source)
        if sources and "Sources:" not in answer:
            answer += "\n\nSources:\n" + "\n".join(f"- {source}" for source in sources)

    print(answer)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--create":
        # Optionally accept a directory path after --create
        if len(sys.argv) > 2:
            create_vector_store(directory_path=sys.argv[2])
        else:
            create_vector_store()
    else:
        main()
