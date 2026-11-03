"""
# 93. Async Ollama RAG Fusion

| Description | Demonstrates asynchronous RAG fusion with Ollama. |
|-------------|--------------------------------------------------|
| Model       | llama3.1 (Ollama) via init_chat_model            |
| Provider    | Ollama                                          |
| Embeddings  | OllamaEmbeddings (nomic-embed-text)             |
| Vector Store| FAISS (in-memory)                               |
"""

import asyncio
import os
from typing import List

from langchain.chat_models import init_chat_model
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings

MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.1")
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def create_vectorstore() -> FAISS:
    """Create an in-memory FAISS vector store with sample documents."""
    documents = [
        Document(
            page_content="The sky is blue and clear on a sunny day.",
            metadata={"source": "weather"},
        ),
        Document(
            page_content="Ollama runs large language models locally on your machine.",
            metadata={"source": "ollama"},
        ),
        Document(
            page_content="RAG fusion combines multiple retrieval queries to improve results.",
            metadata={"source": "rag"},
        ),
        Document(
            page_content="Asynchronous programming in Python uses asyncio to manage concurrency.",
            metadata={"source": "python"},
        ),
        Document(
            page_content="Reciprocal rank fusion is a method for merging ranked lists.",
            metadata={"source": "rag"},
        ),
    ]
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    return FAISS.from_documents(documents, embeddings)


async def generate_queries(chat_model, original_query: str) -> List[str]:
    """Generate multiple search queries from the original query."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that generates diverse search queries for retrieval. "
                "Generate exactly {num_queries} queries, one per line, no numbering.",
            ),
            ("human", "Original query: {query}"),
        ]
    )
    chain = prompt | chat_model | StrOutputParser()
    num_queries = 4
    response = await chain.ainvoke(
        {"query": original_query, "num_queries": num_queries}
    )
    queries = [line.strip() for line in response.splitlines() if line.strip()]
    return queries[:num_queries]


async def retrieve_documents(
    queries: List[str], vectorstore: FAISS
) -> List[List[Document]]:
    """Retrieve documents for each query asynchronously."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    tasks = [asyncio.to_thread(retriever.invoke, query) for query in queries]
    return await asyncio.gather(*tasks)


def reciprocal_rank_fusion(
    docs_lists: List[List[Document]], k: int = 60
) -> List[Document]:
    """Fuse retrieved documents using reciprocal rank fusion."""
    fused_scores = {}
    for docs in docs_lists:
        for rank, doc in enumerate(docs):
            doc_id = doc.page_content  # simple key for demo
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0.0
            fused_scores[doc_id] += 1.0 / (rank + k)

    ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    doc_map = {doc.page_content: doc for docs in docs_lists for doc in docs}
    return [doc_map[doc_id] for doc_id, _ in ranked]


async def generate_final_answer(
    chat_model, query: str, documents: List[Document]
) -> str:
    """Generate the final answer using the fused documents."""
    context = "\n\n".join(doc.page_content for doc in documents)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer the question using only the provided context.",
            ),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ]
    )
    chain = prompt | chat_model | StrOutputParser()
    return await chain.ainvoke({"context": context, "question": query})


async def main() -> None:
    """Run the async RAG fusion example."""
    vectorstore = await asyncio.to_thread(create_vectorstore)

    chat_model = init_chat_model(
        MODEL_NAME, model_provider="ollama", base_url=OLLAMA_BASE_URL
    )

    original_query = "How does RAG fusion work?"
    print(f"Original query: {original_query}")

    queries = await generate_queries(chat_model, original_query)
    print(f"Generated queries: {queries}")

    docs_lists = await retrieve_documents(queries, vectorstore)
    print(
        f"Retrieved {sum(len(docs) for docs in docs_lists)} "
        f"documents across {len(docs_lists)} queries"
    )

    fused_docs = reciprocal_rank_fusion(docs_lists)
    print(f"Fused to {len(fused_docs)} documents")

    answer = await generate_final_answer(chat_model, original_query, fused_docs)
    print(f"\nFinal answer:\n{answer}")


if __name__ == "__main__":
    asyncio.run(main())
