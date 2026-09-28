"""RAG example: retrieve and generate with graceful empty handling."""

import os
from typing import Callable, List

from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.llms.base import BaseLLM
from langchain.schema import BaseRetriever, Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS


def format_docs(docs: List[Document]) -> str:
    """Format retrieved documents into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(
    retriever: BaseRetriever, llm: BaseLLM
) -> Callable[[str], str]:
    """Build a RAG chain that retrieves documents and generates an answer.

    Args:
        retriever: A retriever with a ``get_relevant_documents`` method.
        llm: A language model instance used to generate the final answer.

    Returns:
        A function that accepts a query string and returns the generated answer.
    """

    def rag_chain(query: str) -> str:
        docs = retriever.get_relevant_documents(query)
        if not docs:
            return "I couldn't find any relevant information to answer your question."

        context = format_docs(docs)
        prompt = (
            f"Based on the following context, answer the question.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        )
        return llm(prompt)

    return rag_chain


def load_and_index_documents(file_path: str) -> BaseRetriever:
    """Load documents, split them, and create a retriever from the vector store."""
    loader = TextLoader(file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore.as_retriever()


def main() -> None:
    # Load, split, and index documents
    retriever = load_and_index_documents("data.txt")

    # Initialize LLM
    llm = OpenAI(temperature=0)

    # Build the RAG chain
    rag_chain = build_rag_chain(retriever, llm)

    # Query
    query = "What is the capital of France?"
    response = rag_chain(query)
    print(response)


if __name__ == "__main__":
    main()
