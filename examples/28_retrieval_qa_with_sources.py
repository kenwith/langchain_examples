"""
Retrieval QA with Sources

This example demonstrates how to build a retrieval-augmented question-answering
pipeline that returns both the generated answer and the source documents used
to generate it, giving users greater transparency.

| Aspect          | Description                                           |
|-----------------|-------------------------------------------------------|
| Chat model      | Any model supported by `init_chat_model`              |
| Embeddings      | OpenAI `text-embedding-3-small` (requires API key)    |
| Vector store    | FAISS (in-memory)                                    |
| Retrieval       | Similarity search (top-k)                             |
| Output          | Answer text + list of source documents                |

To run:

    export OPENAI_API_KEY=...
    export CHAT_MODEL="openai:gpt-4o-mini"   # optional, default
    python examples/28_retrieval_qa_with_sources.py
"""

import os

from langchain.chat_models import init_chat_model
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings


def build_retriever():
    """Create an in-memory FAISS retriever from a few sample documents."""
    docs = [
        Document(
            page_content=(
                "LangChain is a framework for developing applications powered by "
                "language models. It provides standard interfaces for chains, "
                "agents, and retrieval-augmented generation."
            ),
            metadata={"source": "langchain-overview.txt", "page": 1},
        ),
        Document(
            page_content=(
                "The RetrievalQA chain is a classic LangChain pattern: first retrieve "
                "relevant document chunks, then pass them to a chat model to generate "
                "an answer grounded in those chunks."
            ),
            metadata={"source": "retrieval-qa-pattern.txt", "page": 1},
        ),
        Document(
            page_content=(
                "Returning source documents alongside answers helps users verify the "
                "grounding of a model's response, reducing hallucination and improving "
                "trust in RAG applications."
            ),
            metadata={"source": "transparency-in-rag.txt", "page": 2},
        ),
        Document(
            page_content=(
                "FAISS is a library for efficient similarity search. In this example, "
                "we store text embeddings in a FAISS vector store and retrieve the "
                "top-k most relevant chunks for a query."
            ),
            metadata={"source": "faiss-notes.txt", "page": 1},
        ),
    ]
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 2})


def ask_with_sources(question, retriever, llm):
    """Retrieve context, ask the LLM, and return (answer, source_documents)."""
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer the user's question using only "
                "the provided context. If the context does not contain the answer, "
                "say you don't know.",
            ),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ]
    )
    messages = prompt.format_messages(context=context, question=question)
    answer = llm.invoke(messages).content
    return answer, docs


def print_sources(sources):
    """Print the source documents as a simple README-style table."""
    print("\nSources:")
    print("| # | Source | Content |")
    print("|---|--------|---------|")
    for i, doc in enumerate(sources, start=1):
        content = doc.page_content.replace("|", "\\|")
        source = doc.metadata.get("source", "unknown")
        print(f"| {i} | {source} | {content} |")


def main():
    """Run the retrieval QA with sources demo."""
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Please set the OPENAI_API_KEY environment variable.")

    model = os.getenv("CHAT_MODEL", "openai:gpt-4o-mini")
    llm = init_chat_model(model=model, temperature=0)

    retriever = build_retriever()

    question = "Why should RAG applications return source documents?"
    answer, sources = ask_with_sources(question, retriever, llm)

    print(f"Question: {question}\n")
    print(f"Answer: {answer}")
    print_sources(sources)


if __name__ == "__main__":
    main()
