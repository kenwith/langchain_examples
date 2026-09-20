#!/usr/bin/env python3
"""RAG example with source attribution.

This example builds a simple retrieval-augmented generation (RAG) pipeline
that returns the sources used to answer a question.
"""

import os

from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter


def main():
    # Check for OpenAI API key
    if "OPENAI_API_KEY" not in os.environ:
        print("Please set the OPENAI_API_KEY environment variable.")
        return

    # Create a small corpus of documents
    corpus = [
        Document(
            page_content="The capital of France is Paris. It is known for the Eiffel Tower.",
            metadata={"source": "france.txt", "page": 1},
        ),
        Document(
            page_content="Paris is the largest city in France, with a population of about 2 million.",
            metadata={"source": "france.txt", "page": 2},
        ),
        Document(
            page_content="Germany's capital is Berlin. Berlin is also the seat of the German government.",
            metadata={"source": "germany.txt", "page": 1},
        ),
        Document(
            page_content="Italy's capital is Rome, which is home to the Colosseum and the Vatican.",
            metadata={"source": "italy.txt", "page": 1},
        ),
    ]

    # Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    docs = text_splitter.split_documents(corpus)

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(docs, embeddings)

    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # Create LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Create QA chain that returns source documents
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        verbose=False,
    )

    # Ask a question
    question = "What is the capital of France?"
    print(f"Query: {question}\n")

    result = qa.invoke({"query": question})

    # Print answer
    print("Answer:", result["result"])
    print("\nSources used:")

    # Print unique sources
    seen = set()
    for doc in result.get("source_documents", []):
        source = doc.metadata.get("source")
        if source not in seen:
            seen.add(source)
            print(f"- {source}")
            # Optionally print a snippet
            snippet = doc.page_content[:100].replace("\n", " ")
            print(f"  Snippet: {snippet}...")
    print()


if __name__ == "__main__":
    main()
