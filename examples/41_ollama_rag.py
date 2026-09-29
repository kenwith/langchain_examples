"""Local RAG with Ollama and FAISS.

This example demonstrates how to build a retrieval-augmented generation (RAG)
pipeline using a local LLM served by Ollama, OllamaEmbeddings for embeddings,
and FAISS for vector storage and retrieval.
"""

import os
import tempfile

from langchain.chat_models import init_chat_model
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# -----------------------------------------------------------------------------
# Example 41: Ollama RAG
# -----------------------------------------------------------------------------
# | Component      | Choice                                              |
# |----------------|-----------------------------------------------------|
# | LLM            | init_chat_model(..., model_provider='ollama')       |
# | Embeddings     | OllamaEmbeddings                                    |
# | Vector Store   | FAISS                                               |
# -----------------------------------------------------------------------------


def load_documents(file_path: str):
    """Load documents from a local text file."""
    loader = TextLoader(file_path)
    return loader.load()


def split_documents(documents):
    """Split documents into chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)


def build_vector_store(chunks, embeddings):
    """Create a FAISS vector store from document chunks."""
    return FAISS.from_documents(chunks, embeddings)


def format_docs(docs):
    """Format retrieved documents for the prompt."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(vector_store, llm):
    """Build a RAG chain using LCEL."""
    retriever = vector_store.as_retriever()
    template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
    prompt = PromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    return rag_chain


def main():
    """Run the local RAG example."""
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model_name = os.getenv("OLLAMA_MODEL", "llama3")
    doc_path = os.getenv("DOC_PATH", "")

    if not doc_path:
        # Create a temporary sample document for demonstration purposes.
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(
                "LangChain is a framework for developing applications powered by "
                "language models.\n"
                "Ollama allows you to run large language models locally on your "
                "machine.\n"
                "FAISS is a library for efficient similarity search and clustering "
                "of dense vectors.\n"
            )
            doc_path = f.name

    llm = init_chat_model(
        model=model_name,
        model_provider="ollama",
        base_url=ollama_base_url,
    )
    embeddings = OllamaEmbeddings(
        model=model_name,
        base_url=ollama_base_url,
    )

    documents = load_documents(doc_path)
    chunks = split_documents(documents)
    vector_store = build_vector_store(chunks, embeddings)
    rag_chain = build_rag_chain(vector_store, llm)

    question = "What is LangChain and how does Ollama relate to it?"
    print(f"Question: {question}\n")
    answer = rag_chain.invoke(question)
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
