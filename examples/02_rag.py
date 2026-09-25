"""RAG example: retrieve and generate with graceful empty handling."""

import os

from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS


def prepare_response(retriever, query, llm):
    """Generate a response from retriever and LLM, handling empty retrieval."""
    docs = retriever.get_relevant_documents(query)
    if not docs:
        return "I couldn't find any relevant information to answer your question."

    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = (
        f"Based on the following context, answer the question.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )
    return llm(prompt)


def main():
    # Load documents
    loader = TextLoader("data.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    # Create vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)
    retriever = vectorstore.as_retriever()

    # Initialize LLM
    llm = OpenAI(temperature=0)

    # Query
    query = "What is the capital of France?"
    response = prepare_response(retriever, query, llm)
    print(response)


if __name__ == "__main__":
    main()
