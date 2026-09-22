"""
Retrieval-Augmented Generation (RAG) pipeline example.

This script demonstrates a complete RAG workflow:
1. Load documents from a local directory.
2. Split the documents into smaller chunks for precise retrieval.
3. Generate embeddings for each chunk and index them in a Chroma vector store.
4. Use the vector store as a retriever to fetch relevant chunks for a query.
5. Pass the retrieved context to an LLM to generate an answer grounded in the documents.

The pipeline combines indexing (steps 1-3) and querying (steps 4-5). It is designed
to be a minimal but functional starting point for building RAG applications.
"""

import os
from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Path to the directory containing your document files
DATA_PATH = "data"

def load_documents(directory: str = DATA_PATH) -> List[Document]:
    """
    Load documents from the specified directory.

    This function scans the given directory for text files and loads their
    content into a list of Document objects. If the directory does not exist
    or contains no text files, it falls back to a hardcoded default document.

    Args:
        directory (str): Path to the directory containing text files.
                         Defaults to DATA_PATH.

    Returns:
        List[Document]: A list of Document objects loaded from the directory.

    Raises:
        FileNotFoundError: If the directory does not exist and no fallback
                           document can be loaded.
    """
    documents = []

    # Check if the directory exists
    if os.path.isdir(directory):
        # Iterate through all files in the directory
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                filepath = os.path.join(directory, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as input_file:
                        text = input_file.read()
                    # Create a Document with metadata (source file name)
                    documents.append(Document(page_content=text, metadata={"source": filename}))
                except UnicodeDecodeError:
                    # Skip files that cannot be decoded, but log a warning
                    print(f"Warning: Could not decode file {filename}. Skipping.")
                    continue
    else:
        # Fallback if the directory doesn't exist
        print(f"Warning: Directory '{directory}' not found. Using built-in default document.")
        # A small embedded default document to keep the script functional
        fallback_text = (
            "LangChain is a framework for developing applications powered by language models. "
            "It provides modular components and integrations to build complex workflows. "
            "This is a fallback document used when no external data is available."
        )
        documents.append(Document(page_content=fallback_text, metadata={"source": "built-in"}))

    # If no documents were loaded (empty directory or no text files), use fallback
    if not documents:
        print("No text files found. Using built-in default document.")
        fallback_text = (
            "LangChain is a framework for developing applications powered by language models. "
            "It provides modular components and integrations to build complex workflows. "
            "This is a fallback document used when no external data is available."
        )
        documents.append(Document(page_content=fallback_text, metadata={"source": "built-in"}))

    return documents

def load_and_split_documents(directory: str = DATA_PATH, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Load documents from the specified directory and split them into chunks.

    This helper combines document loading and splitting for convenience and
    reusability. It uses RecursiveCharacterTextSplitter with configurable
    chunk size and overlap.

    Args:
        directory (str): Path to the directory containing text files.
                         Defaults to DATA_PATH.
        chunk_size (int): Maximum size of each chunk. Defaults to 1000.
        chunk_overlap (int): Number of characters to overlap between chunks.
                             Defaults to 200.

    Returns:
        List[Document]: A list of Document chunks ready for embedding.
    """
    documents = load_documents(directory)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)

def build_vectorstore(document_chunks: List[Document], embedding_model: OpenAIEmbeddings, persist_directory: str = "./chroma_db") -> Chroma:
    """
    Create and persist a Chroma vector store from document chunks.

    This helper encapsulates the vector store creation logic so it can be
    reused and tested independently. It generates embeddings for each document
    chunk and stores them in a Chroma index, then persists the index to disk.

    Args:
        document_chunks (List[Document]): Document chunks to index.
        embedding_model (OpenAIEmbeddings): Embedding model used to vectorize chunks.
        persist_directory (str): Directory where the Chroma index will be persisted.
                                 Defaults to "./chroma_db".

    Returns:
        Chroma: The created and persisted vector store.
    """
    vector_store = Chroma.from_documents(
        documents=document_chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    vector_store.persist()
    return vector_store

def main():
    """
    Main execution function for the RAG example.

    This function runs the full RAG pipeline:
    - Load and split documents.
    - Create an embedding model.
    - Index the document chunks in a Chroma vector store.
    - Retrieve relevant chunks for a sample query.
    - Generate an answer using an LLM with the retrieved context.

    The vector store is persisted locally under ./chroma_db, allowing the
    index to be reused in later runs without re-indexing.
    """
    # 1. Load and split documents into chunks
    document_chunks = load_and_split_documents()

    # 2. Create embedding model
    embedding_model = OpenAIEmbeddings()

    # 3. Create and index the vector store.
    # Chroma builds an index by computing embeddings for each document chunk and
    # storing them alongside the original text. This enables efficient similarity
    # search later. The index is persisted to disk so it can be reused.
    vector_store = build_vectorstore(
        document_chunks=document_chunks,
        embedding_model=embedding_model,
        persist_directory="./chroma_db"
    )

    # 4. Set up retriever using the vector store's index.
    # The retriever fetches the top k most similar chunks for a given query.
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # 5. Create QA chain
    language_model = OpenAI(temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=language_model,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    # 6. Run a sample query
    query = "What is LangChain?"
    response = qa_chain({"query": query})
    print(f"Answer: {response['result']}")
    print("Sources:")
    for doc in response["source_documents"]:
        print(f"- {doc.metadata.get('source', 'unknown')}")

if __name__ == "__main__":
    main()
