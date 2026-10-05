"""Example 54: Ollama Async RAG.

Description: This example demonstrates asynchronous retrieval-augmented
generation (RAG) using Ollama as the local LLM provider. It builds an
in-memory vector store from sample documents, retrieves relevant context,
and generates an answer with an Ollama chat model.

Before running, ensure Ollama is installed and running locally, and that the
desired model (e.g., llama3) is pulled. Optionally set OLLAMA_BASE_URL to
override the default endpoint.
"""

import asyncio
import os

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents() -> list[str]:
    """Return a small set of sample documents for the RAG demo."""
    return [
        "The sky is blue because of Rayleigh scattering.",
        "Ollama allows running large language models locally.",
        "Retrieval-augmented generation combines retrieval with generation.",
        "LangChain provides tools for building applications with LLMs.",
        "Asynchronous code can improve performance for I/O-bound tasks.",
        "Ollama supports embedding models for semantic search.",
        "RAG pipelines retrieve relevant context before generating an answer.",
    ]


async def build_vectorstore() -> InMemoryVectorStore:
    """Split documents, embed them, and return an in-memory vector store."""
    texts = load_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    chunks = splitter.split_text("\n\n".join(texts))

    embeddings = OllamaEmbeddings(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )

    vectorstore = await InMemoryVectorStore.afrom_texts(
        chunks,
        embedding=embeddings,
    )
    return vectorstore


async def answer_question(question: str, vectorstore: InMemoryVectorStore) -> str:
    """Retrieve relevant context and generate an answer using Ollama."""
    llm = init_chat_model(
        "llama3",
        model_provider="ollama",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template(
        "Answer the question using only the provided context.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return await rag_chain.ainvoke(question)


async def main() -> None:
    """Run the async RAG demo."""
    print("Building vector store...")
    vectorstore = await build_vectorstore()

    question = "What is retrieval-augmented generation?"
    print(f"Question: {question}")

    answer = await answer_question(question, vectorstore)
    print(f"Answer: {answer}")


if __name__ == "__main__":
    asyncio.run(main())
