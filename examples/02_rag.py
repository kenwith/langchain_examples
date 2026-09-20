import argparse
import os

from langchain.chains import LLMChain
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

# Set DATA_PATH to an environment variable or default to "data"
DATA_PATH = os.getenv("DATA_PATH", "data")

# Configurable chunking parameters for experimentation
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 0

# Optional cache path for vector store reuse
VECTORSTORE_CACHE_PATH = os.getenv("VECTORSTORE_CACHE_PATH")


def format_docs(docs: list[Document]) -> str:
    """Format a list of documents into a single string for prompt injection."""
    return "\n\n".join(doc.page_content for doc in docs)


def load_documents_from_directory(directory_path: str) -> list[Document]:
    """Load all text documents from a given directory.

    Args:
        directory_path: Path to the directory containing text files.

    Returns:
        List of loaded documents.

    Raises:
        ValueError: If the directory does not exist or contains no text files.
    """
    if not os.path.isdir(directory_path):
        raise ValueError(f"Directory '{directory_path}' does not exist.")
    loader = DirectoryLoader(directory_path, loader_cls=TextLoader)
    documents = loader.load()
    if not documents:
        raise ValueError(f"No documents found in directory '{directory_path}'.")
    return documents


def load_documents() -> list[Document]:
    """Load documents from the DATA_PATH directory or fallback to built-in examples.

    If DATA_PATH exists and is a directory, load all text files from it.
    If the directory exists but contains no files, raise a clear error.
    If the directory does not exist, return a small set of built-in example documents.
    """
    if os.path.exists(DATA_PATH):
        return load_documents_from_directory(DATA_PATH)
    else:
        # Fallback built-in document set so the example runs without external files.
        builtin_texts = [
            "LangChain is a framework for developing applications powered by language models.",
            "Retrieval-augmented generation (RAG) combines retrieval of relevant documents with a language model.",
            "RAG helps reduce hallucination by grounding the model on external knowledge.",
            "Vector stores like FAISS allow efficient similarity search over document embeddings.",
            "To use this example, set OPENAI_API_KEY in your environment.",
        ]
        return [Document(page_content=text) for text in builtin_texts]


def build_vectorstore_from_directory(directory_path: str, cache_path: str = None) -> FAISS:
    """Build a FAISS vector store from documents in a directory, with optional caching.

    Args:
        directory_path: Path to the directory containing text files.
        cache_path: If provided, the vector store will be saved to this path and
            loaded from it on subsequent calls if it exists.

    Returns:
        A FAISS vector store built from the documents.
    """
    # Try to load from cache if a cache path is provided and exists
    if cache_path and os.path.exists(cache_path):
        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        vectorstore = FAISS.load_local(cache_path, embeddings)
        return vectorstore

    # Load documents and split them
    documents = load_documents_from_directory(directory_path)
    text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    texts = text_splitter.split_documents(documents)

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore = FAISS.from_documents(texts, embeddings)

    # Save to cache if requested
    if cache_path:
        vectorstore.save_local(cache_path)

    return vectorstore


def build_vectorstore() -> FAISS:
    """Build and return a FAISS vector store from documents.

    Uses the global DATA_PATH and optional VECTORSTORE_CACHE_PATH for caching.
    """
    return build_vectorstore_from_directory(DATA_PATH, VECTORSTORE_CACHE_PATH)


def answer_question(vectorstore: FAISS, query: str) -> str:
    """Answer a query using retrieval-augmented generation.

    Sets up a retriever from the provided vector store, retrieves relevant
    documents, formats them with the `format_docs` helper, and runs a simple
    LLMChain with a custom prompt that includes the formatted context.
    """
    # Set up the retriever and retrieve relevant documents
    retriever = vectorstore.as_retriever()
    docs = retriever.get_relevant_documents(query)
    context = format_docs(docs)

    # Define the prompt template
    prompt = PromptTemplate(
        template="""Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:""",
        input_variables=["context", "question"]
    )

    # Create the LLM chain and run it
    llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(context=context, question=query)


def main(query: str = "What is RAG?") -> str:
    """Build the vector store and return the generated answer for the query."""
    vectorstore = build_vectorstore()
    return answer_question(vectorstore, query)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run retrieval-augmented generation with an optional query.")
    parser.add_argument("query", nargs="?", default="What is RAG?", help="Query to ask (default: %(default)s)")
    args = parser.parse_args()
    print(main(args.query))
