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

    splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore.as_retriever()


def main():
    retriever = build_retriever()

    prompt = ChatPromptTemplate.from_template(
        "Answer the question based on the following context:\n{{ format_docs(docs) }}\n\nQuestion: {{ question }}",
        template_format="jinja2",
        partial_variables={"format_docs": format_docs},
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

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
