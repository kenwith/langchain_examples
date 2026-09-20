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
                    with open(filepath, "r", encoding="utf-8") as f:
                        text = f.read()
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

def main():
    """
    Main execution function for the RAG example.

    Loads documents, splits them into chunks, creates embeddings, stores them
    in a vector database, and sets up a retrieval-based QA pipeline.
    """
    # 1. Load documents
    docs = load_documents()

    # 2. Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)

    # 3. Create embeddings
    embeddings = OpenAIEmbeddings()

    # 4. Create vector store (persist to disk)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    vectorstore.persist()

    # 5. Set up retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 6. Create QA chain
    llm = OpenAI(temperature=0)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    # 7. Run a sample query
    query = "What is LangChain?"
    result = qa({"query": query})
    print(f"Answer: {result['result']}")
    print("Sources:")
    for doc in result["source_documents"]:
        print(f"- {doc.metadata.get('source', 'unknown')}")

if __name__ == "__main__":
    main()
