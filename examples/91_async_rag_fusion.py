"""Async RAG-Fusion pipeline using asyncio.gather and retriever.ainvoke.

This example demonstrates how to combine multiple retrieved document sets
with reciprocal rank fusion (RRF) in an asynchronous LangChain pipeline.
"""

import asyncio
import os

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

DEFAULT_NUM_QUERIES = 3
DEFAULT_RRF_K = 60

QUERY_PROMPT = ChatPromptTemplate.from_template(
    "You are a helpful assistant that generates multiple search queries based on a single input query. "
    "Generate {num_queries} search queries, one per line, related to the following input query:\n{question}"
)

RAG_PROMPT = ChatPromptTemplate.from_template(
    "Answer the following question based only on the provided context. "
    "If the answer cannot be found in the context, say so.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)


async def generate_search_queries(
    question: str,
    model,
    num_queries: int = DEFAULT_NUM_QUERIES,
) -> list[str]:
    """Generate multiple search queries from a single question."""
    chain = QUERY_PROMPT | model | StrOutputParser()
    response = await chain.ainvoke(
        {"question": question, "num_queries": num_queries}
    )
    queries = [line.strip() for line in response.splitlines() if line.strip()]
    return queries[:num_queries]


def reciprocal_rank_fusion(
    retrieved_docs: list[list[Document]],
    k: int = DEFAULT_RRF_K,
) -> list[Document]:
    """Fuse multiple ranked document lists using reciprocal rank fusion."""
    fused_scores = {}
    for docs in retrieved_docs:
        for rank, doc in enumerate(docs):
            doc_id = doc.page_content
            if doc_id not in fused_scores:
                fused_scores[doc_id] = {"score": 0.0, "doc": doc}
            fused_scores[doc_id]["score"] += 1.0 / (k + rank + 1)

    sorted_docs = sorted(
        fused_scores.values(),
        key=lambda item: item["score"],
        reverse=True,
    )
    return [item["doc"] for item in sorted_docs]


async def retrieve_and_fuse(
    question: str,
    retriever,
    model,
    num_queries: int = DEFAULT_NUM_QUERIES,
    k: int = DEFAULT_RRF_K,
) -> list[Document]:
    """Generate search queries, retrieve documents concurrently, and fuse."""
    queries = await generate_search_queries(question, model, num_queries)
    retrieval_tasks = [retriever.ainvoke(query) for query in queries]
    retrieved_docs = await asyncio.gather(*retrieval_tasks)
    return reciprocal_rank_fusion(retrieved_docs, k=k)


def format_docs(docs: list[Document]) -> str:
    """Format documents for context."""
    return "\n\n".join(doc.page_content for doc in docs)


async def generate_answer(
    question: str,
    fused_docs: list[Document],
    model,
) -> str:
    """Generate a final answer from fused documents."""
    chain = RAG_PROMPT | model | StrOutputParser()
    context = format_docs(fused_docs)
    return await chain.ainvoke({"question": question, "context": context})


async def main(question: str) -> None:
    """Run the async RAG-Fusion pipeline."""
    model = init_chat_model(
        os.getenv("CHAT_MODEL", "gpt-4o-mini"),
        model_provider=os.getenv("CHAT_MODEL_PROVIDER", "openai"),
        temperature=0,
    )

    embeddings = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    )

    texts = [
        "LangChain is a framework for developing applications powered by language models.",
        "RAG-Fusion combines multiple retrieval queries using reciprocal rank fusion.",
        "Asyncio allows concurrent execution of asynchronous tasks in Python.",
        "Retrievers in LangChain can be invoked asynchronously with `ainvoke`.",
        "Reciprocal rank fusion scores documents by the inverse of their rank.",
    ]

    vectorstore = InMemoryVectorStore.from_texts(texts, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    fused_docs = await retrieve_and_fuse(question, retriever, model)
    answer = await generate_answer(question, fused_docs, model)

    print(f"Question: {question}")
    print(f"Fused documents: {len(fused_docs)}")
    for i, doc in enumerate(fused_docs, start=1):
        print(f"{i}. {doc.page_content}")
    print(f"Answer: {answer}")


if __name__ == "__main__":
    asyncio.run(main("How does RAG-Fusion work?"))
