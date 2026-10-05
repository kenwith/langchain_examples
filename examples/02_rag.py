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
    try:
        loader = TextLoader(DATA_FILE)
        documents = loader.load()
    except FileNotFoundError:
        print(f"Error: Data file '{DATA_FILE}' not found.", file=sys.stderr)
        return []
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_documents(documents)


def create_vectorstore():
    """Create and persist the vector store from documents."""
    docs = load_documents()
    if not docs:
        print("No documents found to index. Exiting.", file=sys.stderr)
        return
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        docs, embeddings, persist_directory=PERSIST_DIR
    )
    vectorstore.persist()
    print(f"Vector store created at {PERSIST_DIR}")


def format_docs(docs):
    """Format a list of documents into a single string for context."""
    return "\n\n".join(doc.page_content for doc in docs)


def retrieve_context(retriever, query):
    """Retrieve relevant documents and format them as context.

    If no documents are found, return a fallback message so the LLM
    knows there is no relevant context to use.
    """
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant documents found in the knowledge base."
    return format_docs(docs)


def main():
    """Run the RAG pipeline."""
    if not os.path.exists(PERSIST_DIR):
        print(
            f"Vector store not found at '{PERSIST_DIR}'. "
            f"Run `python {os.path.basename(__file__)} --create` to create it.",
            file=sys.stderr,
        )
        return

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

    # Build the RAG chain using LCEL.
    # 1. Retrieval step: fetch relevant documents from the vector store
    #    based on the user's query, then format them into a context string.
    #    If no documents are retrieved, a fallback message is used.
    # 2. Generation step: pass the context and the original question to the
    #    LLM via the prompt template, and generate the final answer.
    rag_chain = (
        RunnablePassthrough.assign(
            context=lambda x: retrieve_context(retriever, x["input"])
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
