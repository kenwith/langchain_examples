"""
RAG (Retrieval-Augmented Generation) Example

Demonstrates: Document loading, splitting, embedding, vector store, retrieval, generation
Provider-agnostic using init_chat_model
"""
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

load_dotenv()


def get_model():
    model_name = os.getenv("LANGCHAIN_MODEL", "openai/gpt-4o-mini")
    return init_chat_model(model_name)


def get_embeddings():
    """Use local embeddings (no API key needed)"""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def create_sample_documents():
    """Create sample documents for demonstration"""
    return [
        Document(
            page_content="LangChain is a framework for developing applications powered by language models. "
                         "It provides modular components for building LLM applications including prompt templates, "
                         "output parsers, and chains.",
            metadata={"source": "langchain_intro", "topic": "framework"}
        ),
        Document(
            page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs. "
                         "It extends LangChain with graph-based workflows, enabling cycles, branching, "
                         "and human-in-the-loop interactions.",
            metadata={"source": "langgraph_intro", "topic": "workflows"}
        ),
        Document(
            page_content="RAG (Retrieval-Augmented Generation) combines information retrieval with text generation. "
                         "It retrieves relevant documents from a knowledge base and uses them as context "
                         "for generating more accurate responses.",
            metadata={"source": "rag_intro", "topic": "rag"}
        ),
        Document(
            page_content="Vector stores like FAISS, Chroma, and Pinecone enable semantic search by storing "
                         "document embeddings. They support similarity search, max marginal relevance, "
                         "and metadata filtering.",
            metadata={"source": "vector_stores", "topic": "vector_stores"}
        ),
        Document(
            page_content="Prompt engineering is the practice of designing effective prompts for LLMs. "
                         "Key techniques include few-shot prompting, chain-of-thought, and structured outputs "
                         "using Pydantic models.",
            metadata={"source": "prompt_engineering", "topic": "prompting"}
        ),
    ]


def build_vector_store(documents, embeddings):
    """Build FAISS vector store from documents"""
    print("Building vector store...")
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]
    vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    return vector_store


def basic_rag_chain():
    """Basic RAG chain: retrieve -> format -> generate"""
    print("=== Basic RAG Chain ===")

    documents = create_sample_documents()
    embeddings = get_embeddings()
    vector_store = build_vector_store(documents, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    model = get_model()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the question using only the provided context. "
                   "If the answer isn't in the context, say 'I don't know based on the provided context.'"),
        ("user", "Context:\n{context}\n\nQuestion: {question}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    questions = [
        "What is LangChain?",
        "What does LangGraph enable?",
        "What is RAG?",
        "What are vector stores used for?",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        result = rag_chain.invoke(q)
        print(f"A: {result}")

    return rag_chain


def rag_with_metadata_filter():
    """RAG with metadata filtering"""
    print("\n=== RAG with Metadata Filter ===")

    documents = create_sample_documents()
    embeddings = get_embeddings()
    vector_store = build_vector_store(documents, embeddings)

    # Filter by topic
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 2, "filter": {"topic": "rag"}}
    )

    model = get_model()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer using only the provided context."),
        ("user", "Context:\n{context}\n\nQuestion: {question}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    result = chain.invoke("How does retrieval work in RAG?")
    print(f"Filtered result: {result}")
    return result


def rag_with_sources():
    """RAG that returns sources alongside answer"""
    print("\n=== RAG with Sources ===")

    documents = create_sample_documents()
    embeddings = get_embeddings()
    vector_store = build_vector_store(documents, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    model = get_model()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the question using the provided context. "
                   "Cite sources using [source_name] format."),
        ("user", "Context:\n{context}\n\nQuestion: {question}"),
    ])

    def format_docs_with_sources(docs):
        return "\n\n".join(
            f"[{doc.metadata['source']}] {doc.page_content}"
            for doc in docs
        )

    chain = (
        {"context": retriever | format_docs_with_sources, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    result = chain.invoke("What are the key components of LangChain?")
    print(f"Result with sources: {result}")
    return result


if __name__ == "__main__":
    basic_rag_chain()
    rag_with_metadata_filter()
    rag_with_sources()
    print("\nAll RAG examples completed!")