"""
Example 52: Async RAG with init_chat_model
===========================================

This example demonstrates how to build a RAG (Retrieval-Augmented Generation)
chain and run it asynchronously using `ainvoke` and `astream`.

The chain uses:
- `init_chat_model` for a provider-agnostic chat model.
- An in-memory vector store with OpenAI embeddings.
- A simple prompt that includes retrieved context.

| # | Example                         | Description                                   |
|---|---------------------------------|-----------------------------------------------|
| 52| async_rag.py                    | Async RAG with ainvoke and astream            |

Prerequisites:
- Set `OPENAI_API_KEY` in your environment.
- Install: `pip install langchain langchain-openai langchain-core`
"""

import asyncio
import os

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings


def build_chain():
    """Build a RAG chain with an in-memory vector store and async retrieval."""
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Please set OPENAI_API_KEY in your environment.")

    # Small corpus for the in-memory vector store.
    documents = [
        Document(page_content="LangChain is a framework for developing applications powered by language models."),
        Document(page_content="RAG stands for Retrieval-Augmented Generation."),
        Document(page_content="Async support allows concurrent execution of chains."),
    ]

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = InMemoryVectorStore.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    async def retrieve_and_format(inputs):
        docs = await retriever.ainvoke(inputs["question"])
        return {
            "context": format_docs(docs),
            "question": inputs["question"],
        }

    chat_model = init_chat_model(
        model="gpt-4o-mini",
        provider="openai",
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the question based only on the following context:\n\n{context}",
            ),
            ("human", "{question}"),
        ]
    )

    chain = (
        RunnableLambda(retrieve_and_format)
        | prompt
        | chat_model
        | StrOutputParser()
    )
    return chain


async def main():
    """Run the RAG chain with ainvoke and astream."""
    print("Building RAG chain...")
    chain = build_chain()

    question = "What is RAG?"
    print(f"\nQuestion: {question}\n")

    print("--- ainvoke ---")
    answer = await chain.ainvoke({"question": question})
    print(answer)

    print("\n--- astream ---")
    async for chunk in chain.astream({"question": question}):
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
