"""
Example 88: Async Contextual Compression RAG

This example demonstrates an asynchronous retrieval-augmented generation (RAG)
pipeline with a contextual compression retriever. The retriever uses a
provider-agnostic chat model to compress retrieved documents before they are
used as context by an answer-generation chain.
"""

import asyncio
import os
from typing import List

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents() -> List[Document]:
    """Load a small set of sample documents into memory."""
    return [
        Document(
            page_content=(
                "LangChain is a framework designed to simplify the development "
                "of applications with language models. It provides standard, "
                "composable interfaces for chains, agents, and retrievers."
            ),
            metadata={"source": "langchain-overview.txt"},
        ),
        Document(
            page_content=(
                "Retrievers in LangChain abstract the process of fetching relevant "
                "documents. A retriever can wrap a vector store, a web search engine, "
                "or any other data source that returns documents based on a query."
            ),
            metadata={"source": "retrievers.txt"},
        ),
        Document(
            page_content=(
                "Contextual compression selectively narrows the content of retrieved "
                "documents. A compressor can remove irrelevant parts while preserving "
                "the pieces that answer a given query."
            ),
            metadata={"source": "compression.txt"},
        ),
        Document(
            page_content=(
                "The LCEL (LangChain Expression Language) interface enables the "
                "construction of chains with a familiar syntax. Runnable objects can "
                "be composed with the pipe operator and support both synchronous and "
                "asynchronous execution."
            ),
            metadata={"source": "lcel.txt"},
        ),
    ]


def get_chat_model():
    """Create a provider-agnostic chat model from environment variables."""
    return init_chat_model(
        os.getenv("CHAT_MODEL", "gpt-4o-mini"),
        model_provider=os.getenv("CHAT_MODEL_PROVIDER", "openai"),
        temperature=0,
    )


def get_embeddings():
    """Create a provider-agnostic embedding model from environment variables."""
    return init_embeddings(
        os.getenv("EMBEDDING_MODEL_PROVIDER", "openai"),
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
    )


def build_retriever():
    """Build a contextual compression retriever over an in-memory vector store."""
    documents = load_documents()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    splits = text_splitter.split_documents(documents)
    vector_store = InMemoryVectorStore.from_documents(splits, get_embeddings())
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    compressor = LLMChainExtractor.from_llm(get_chat_model())
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )


def format_docs(documents: List[Document]) -> str:
    """Join a list of document contents into a single context string."""
    return "\n\n".join(document.page_content for document in documents)


async def main() -> None:
    """Run the asynchronous RAG pipeline and print the answer."""
    retriever = build_retriever()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Answer the question using only the provided context."),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ]
    )

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | get_chat_model()
        | StrOutputParser()
    )

    question = "What does LangChain provide?"
    answer = await chain.ainvoke(question)
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
