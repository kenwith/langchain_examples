import os
import sys

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

DATA_FILE = "data.txt"
PERSIST_DIR = "db"


def load_documents():
    """Load and split documents from the data file."""
    loader = TextLoader(DATA_FILE)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_documents(documents)


def create_vectorstore():
    """Create and persist the vector store from documents."""
    docs = load_documents()
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        docs, embeddings, persist_directory=PERSIST_DIR
    )
    vectorstore.persist()
    print(f"Vector store created at {PERSIST_DIR}")


def format_docs(docs):
    """Format a list of documents into a single string for context."""
    return "\n\n".join(doc.page_content for doc in docs)


def main():
    """Run the RAG pipeline."""
    if not os.path.exists(PERSIST_DIR):
        raise FileNotFoundError(
            f"Vector store not found at '{PERSIST_DIR}'. "
            f"Run `python {os.path.basename(__file__)} --create` to create it."
        )

    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR, embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever()

    llm = ChatOpenAI(model="gpt-3.5-turbo")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer questions based on the provided context.",
            ),
            ("human", "Context: {context}\n\nQuestion: {input}"),
        ]
    )

    # Build the RAG chain using LCEL with RunnablePassthrough.assign
    rag_chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["input"]))
        )
        | prompt
        | llm
    )

    query = "What did the president say about Ketanji Brown Jackson?"
    response = rag_chain.invoke({"input": query})
    print(response.content)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--create":
        create_vectorstore()
    else:
        main()
