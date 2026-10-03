"""Example of a Retrieval-Augmented Generation (RAG) pipeline.

This script builds a simple RAG chain: it retrieves relevant document chunks
from a vector store and passes them as context to an LLM to answer a question.
"""

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter


def format_docs(docs):
    """Join a list of documents into a single string for context."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_retriever():
    # Sample documents - replace with your own data source
    documents = [
        Document(page_content="LangChain is a framework for developing applications powered by language models."),
        Document(page_content="RAG stands for Retrieval-Augmented Generation."),
        Document(page_content="LangChain provides modular components for building RAG pipelines."),
    ]

    # Split documents into smaller chunks for more precise retrieval
    splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    chunks = splitter.split_documents(documents)

    # Generate embeddings and store them in a FAISS vector index
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Return a retriever that can fetch relevant chunks for a query
    return vectorstore.as_retriever()


def main():
    # Build the retriever that will fetch relevant context
    retriever = build_retriever()

    # Define the prompt template. The docs are formatted by format_docs before being inserted.
    prompt = ChatPromptTemplate.from_template(
        "Answer the question based on the following context:\n{{ format_docs(docs) }}\n\nQuestion: {{ question }}",
        template_format="jinja2",
        partial_variables={"format_docs": format_docs},
    )

    # Initialize the language model (uses OpenAI API key from environment)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Construct the RAG chain:
    # 1. Retrieve relevant documents for the input question and pass the question through.
    # 2. Format the retrieved docs and the question into the prompt.
    # 3. Generate an answer with the LLM.
    # 4. Parse the output to a plain string.
    rag_chain = (
        {"docs": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    question = "What is RAG?"
    answer = rag_chain.invoke(question)
    print(answer)


if __name__ == "__main__":
    main()
