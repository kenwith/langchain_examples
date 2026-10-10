import os
import sys

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

DATA_FILE = "data.txt"
PERSIST_DIR = "db"

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


def load_documents(directory_path=None, chunk_size=1000, chunk_overlap=0):
    """Load and split documents from a directory or built-in sample texts.

    If a directory path is provided, all text files within that directory are
    loaded. Otherwise, built-in sample texts are used as a fallback.

    Args:
        directory_path: Optional path to a directory containing text files.
        chunk_size: Maximum size of each text chunk.
        chunk_overlap: Number of characters to overlap between chunks.

    Returns:
        A list of document chunks, or an empty list if no documents are found.
    """
    documents = []

    if directory_path:
        documents = load_documents_from_directory(directory_path)
        if not documents:
            print(
                "Falling back to sample texts.",
                file=sys.stderr,
            )
            documents = [
                Document(page_content=text, metadata={"source": "sample"})
                for text in SAMPLE_TEXTS
            ]
    else:
        documents = [
            Document(page_content=text, metadata={"source": "sample"})
            for text in SAMPLE_TEXTS
        ]

    if not documents:
        print("No documents found to load.", file=sys.stderr)
        return []

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)


def create_vectorstore(directory_path=None):
    """Create and persist the vector store from documents.

    Args:
        directory_path: Optional path to a directory containing text files.
            If not provided, sample texts are used.
    """
    docs = load_documents(directory_path=directory_path)
    if not docs:
        print("No documents found to index. Exiting.", file=sys.stderr)
        return
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        docs, embeddings, persist_directory=PERSIST_DIR
    )
    vectorstore.persist()
    print(f"Vector store created at {PERSIST_DIR}")


def format_docs(docs):
    """Format a list of documents into a single string for use as context.

    This helper keeps prompt construction readable and reusable by
    centralizing the conversion of retrieved documents into a context block.
    """
    return "\n\n".join(doc.page_content for doc in docs)


def retrieve_context(retriever, query):
    """Retrieve relevant documents and format them as context.

    If no documents are found, return a fallback message so the LLM
    knows there is no relevant context to use.
    """
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant documents found in the knowledge base."
    return format_docs(docs)


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

    retriever = vectorstore.as_retriever()

    llm = ChatOpenAI(model="gpt-3.5-turbo")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer questions based on the provided context.",
            ),
            ("human", "Context: {context}\n\nQuestion: {input}"),
        ]
    )

    # Build the RAG chain using LCEL.
    # 1. Retrieval step: fetch relevant documents from the vector store
    #    based on the user's query, then format them into a context string.
    #    If no documents are retrieved, a fallback message is used.
    # 2. Generation step: pass the context and the original question to the
    #    LLM via the prompt template, and generate the final answer.
    rag_chain = (
        RunnablePassthrough.assign(
            context=lambda x: retrieve_context(retriever, x["input"])
        )
        | prompt
        | llm
    )

    query = "What did the president say about Ketanji Brown Jackson?"
    response = rag_chain.invoke({"input": query})
    print(response.content)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--create":
        # Optionally accept a directory path after --create
        if len(sys.argv) > 2:
            create_vectorstore(directory_path=sys.argv[2])
        else:
            create_vectorstore()
    else:
        main()
