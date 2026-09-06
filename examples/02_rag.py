"""
RAG (Retrieval-Augmented Generation) Example

Demonstrates: Document loading, splitting, embedding, vector store, retrieval, reranking, generation
Provider-agnostic using init_chat_model
"""
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

load_dotenv()


def get_model():
    model_name = os.getenv("LANGCHAIN_MODEL", "openai/gpt-4o-mini")
    return init_chat_model(model_name)


def get_embeddings():
    """Use local embeddings (no API key needed)"""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def get_reranker():
    """Get cross-encoder reranker for improving retrieval quality"""
    return HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")


def create_sample_documents():
    """Create larger sample documents that will be split by RecursiveCharacterTextSplitter"""
    return [
        Document(
            page_content="""LangChain is a comprehensive framework for developing applications powered by large language models (LLMs). 
It provides modular components for building LLM applications including prompt templates, output parsers, and chains.
The framework supports multiple LLM providers including OpenAI, Anthropic, Google, and local models through Ollama.
LangChain's core philosophy is composability - each component can be used independently or combined into complex workflows.
Key modules include: langchain-core for base abstractions, langchain-community for third-party integrations,
and langchain for high-level chains and agents. The framework also provides tools for memory management,
allowing applications to maintain conversation history and context across multiple interactions.
Developers can build everything from simple chatbots to complex autonomous agents using LangChain's building blocks.""",
            metadata={"source": "langchain_intro", "topic": "framework", "author": "langchain_team"}
        ),
        Document(
            page_content="""LangGraph is a library for building stateful, multi-actor applications with LLMs, extending LangChain with graph-based workflows.
It enables cycles, branching, and human-in-the-loop interactions that are difficult to achieve with linear chains.
LangGraph uses a directed graph structure where nodes represent computation steps and edges define the flow.
This allows for complex patterns like: parallel execution of multiple LLM calls, conditional branching based on outputs,
loops for iterative refinement, and checkpointing for persistence and recovery.
The library integrates seamlessly with LangChain components and supports both synchronous and asynchronous execution.
Common use cases include: multi-agent systems, recursive workflows, human approval gates, and long-running processes.""",
            metadata={"source": "langgraph_intro", "topic": "workflows", "author": "langchain_team"}
        ),
        Document(
            page_content="""RAG (Retrieval-Augmented Generation) combines information retrieval with text generation to produce more accurate and grounded responses.
It retrieves relevant documents from a knowledge base and uses them as context for generating responses, reducing hallucinations.
The typical RAG pipeline consists of: indexing (document loading, splitting, embedding, storing), retrieval (finding relevant chunks),
and generation (synthesizing answer from retrieved context). Advanced RAG techniques include: query expansion, hybrid search (keyword + semantic),
reranking retrieved results, and iterative retrieval. RAG is particularly valuable for domain-specific applications where the LLM
lacks training data, such as internal documentation, legal documents, or technical specifications.
Evaluation metrics for RAG include: faithfulness, answer relevance, context precision, and context recall.""",
            metadata={"source": "rag_intro", "topic": "rag", "author": "research_team"}
        ),
        Document(
            page_content="""Vector stores like FAISS, Chroma, Pinecone, Weaviate, and Qdrant enable semantic search by storing document embeddings.
They support similarity search (cosine, dot product, Euclidean), max marginal relevance (MMR) for diversity,
and metadata filtering for structured queries. FAISS (Facebook AI Similarity Search) is optimized for CPU/GPU similarity search
with various index types: flat (exact), IVF (inverted file), HNSW (hierarchical navigable small world).
Chroma provides a simple API with persistent storage and built-in embedding functions. Pinecone offers managed vector database
with automatic scaling and hybrid search. Choosing a vector store depends on: scale, latency requirements, metadata complexity,
and operational preferences. All support the standard LangChain VectorStore interface for easy swapping.""",
            metadata={"source": "vector_stores", "topic": "vector_stores", "author": "engineering_team"}
        ),
        Document(
            page_content="""Prompt engineering is the practice of designing effective prompts for LLMs to elicit desired behaviors and outputs.
Key techniques include: few-shot prompting (providing examples), chain-of-thought (step-by-step reasoning),
structured outputs using Pydantic models or JSON schemas, system prompts for defining persona and constraints,
and prompt templates for reusable patterns. Advanced techniques: tree-of-thoughts (exploring multiple reasoning paths),
self-consistency (sampling multiple outputs), and automatic prompt optimization (APO).
Best practices: be specific, provide context, use delimiters, specify output format, and iterate based on results.
LangChain's PromptTemplate and ChatPromptTemplate provide variable interpolation, partial formatting,
and composition for building complex prompts programmatically.""",
            metadata={"source": "prompt_engineering", "topic": "prompting", "author": "research_team"}
        ),
    ]


def split_documents(documents):
    """Split documents using RecursiveCharacterTextSplitter for better chunking"""
    print("Splitting documents with RecursiveCharacterTextSplitter...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks


def build_vector_store(documents, embeddings):
    """Build FAISS vector store from documents"""
    print("Building vector store...")
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store


def create_reranking_retriever(vector_store, base_k=10, final_k=3):
    """Create a retriever with cross-encoder reranking"""
    print("Setting up reranking retriever...")
    base_retriever = vector_store.as_retriever(search_kwargs={"k": base_k})
    
    reranker = CrossEncoderReranker(
        model=get_reranker(),
        top_n=final_k
    )
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever
    )
    return compression_retriever


def basic_rag_chain():
    """Basic RAG chain with recursive splitting and reranking"""
    print("=== Basic RAG Chain (with Recursive Splitting + Reranking) ===")

    raw_documents = create_sample_documents()
    documents = split_documents(raw_documents)
    embeddings = get_embeddings()
    vector_store = build_vector_store(documents, embeddings)
    retriever = create_reranking_retriever(vector_store)

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


def rag_with_metadata_aware_retrieval():
    """RAG with metadata-aware retrieval using self-query style filtering"""
    print("\n=== RAG with Metadata-Aware Retrieval ===")

    raw_documents = create_sample_documents()
    documents = split_documents(raw_documents)
    embeddings = get_embeddings()
    vector_store = build_vector_store(documents, embeddings)

    # Metadata-aware retrieval: filter by topic AND author
    def metadata_aware_retriever(query: str, topic: str = None, author: str = None, k: int = 3):
        filter_dict = {}
        if topic:
            filter_dict["topic"] = topic
        if author:
            filter_dict["author"] = author
        
        search_kwargs = {"k": k}
        if filter_dict:
            search_kwargs["filter"] = filter_dict
        
        base_retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
        # Apply reranking on filtered results
        reranker = CrossEncoderReranker(model=get_reranker(), top_n=k)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=base_retriever
        )
        return compression_retriever.invoke(query)

    model = get_model()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer using only the provided context."),
        ("user", "Context:\n{context}\n\nQuestion: {question}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Example: Query only RAG documents from research_team
    print("Querying RAG topic from research_team...")
    retrieved_docs = metadata_aware_retriever(
        "How does retrieval work in RAG?", 
        topic="rag", 
        author="research_team", 
        k=3
    )
    context = format_docs(retrieved_docs)
    result = (prompt | model | StrOutputParser()).invoke({"context": context, "question": "How does retrieval work in RAG?"})
    print(f"Metadata-filtered result: {result}")
    
    # Show what metadata was used
    print(f"\nRetrieved from sources: {[doc.metadata['source'] for doc in retrieved_docs]}")
    print(f"Topics: {[doc.metadata['topic'] for doc in retrieved_docs]}")
    print(f"Authors: {[doc.metadata['author'] for doc in retrieved_docs]}")
    
    return result


def rag_with_reranking_comparison():
    """Demonstrate the effect of reranking by comparing with and without"""
    print("\n=== RAG: With vs Without Reranking ===")

    raw_documents = create_sample_documents()
    documents = split_documents(raw_documents)
    embeddings = get_embeddings()
    vector_store = build_vector_store(documents, embeddings)

    model = get_model()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the question using only the provided context."),
        ("user", "Context:\n{context}\n\nQuestion: {question}"),
    ])

    def format_docs(docs):
        return "\n\n".join(f"[{doc.metadata['source']}] {doc.page_content[:200]}..." for doc in docs)

    # Without reranking - just vector similarity
    basic_retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # With reranking
    reranked_retriever = create_reranking_retriever(vector_store, base_k=10, final_k=3)

    question = "What indexing techniques does FAISS use?"
    
    print(f"Q: {question}\n")
    
    # Basic retrieval
    basic_docs = basic_retriever.invoke(question)
    print("Without reranking (top-3 by vector similarity):")
    for i, doc in enumerate(basic_docs):
        print(f"  {i+1}. [{doc.metadata['source']}] {doc.page_content[:150]}...")
    
    basic_context = format_docs(basic_docs)
    basic_result = (prompt | model | StrOutputParser()).invoke({"context": basic_context, "question": question})
    print(f"\nAnswer: {basic_result}\n")
    
    # Reranked retrieval
    reranked_docs = reranked_retriever.invoke(question)
    print("With reranking (top-10 by vector similarity, reranked to top-3):")
    for i, doc in enumerate(reranked_docs):
        print(f"  {i+1}. [{doc.metadata['source']}] {doc.page_content[:150]}...")
    
    reranked_context = format_docs(reranked_docs)
    reranked_result = (prompt | model | StrOutputParser()).invoke({"context": reranked_context, "question": question})
    print(f"\nAnswer: {reranked_result}")
    
    return {"basic": basic_result, "reranked": reranked_result}


def rag_with_sources_and_metadata():
    """RAG that returns sources alongside answer with full metadata"""
    print("\n=== RAG with Sources and Full Metadata ===")

    raw_documents = create_sample_documents()
    documents = split_documents(raw_documents)
    embeddings = get_embeddings()
    vector_store = build_vector_store(documents, embeddings)
    retriever = create_reranking_retriever(vector_store)

    model = get_model()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the question using the provided context. "
                   "Cite sources using [source_name] format. Include topic and author if relevant."),
        ("user", "Context:\n{context}\n\nQuestion: {question}"),
    ])

    def format_docs_with_full_metadata(docs):
        return "\n\n".join(
            f"[source: {doc.metadata['source']}, topic: {doc.metadata['topic']}, author: {doc.metadata['author']}] {doc.page_content}"
            for doc in docs
        )

    chain = (
        {"context": retriever | format_docs_with_full_metadata, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    result = chain.invoke("What are the key components of LangChain and how does LangGraph extend it?")
    print(f"Result with full metadata sources: {result}")
    return result


if __name__ == "__main__":
    basic_rag_chain()
    rag_with_metadata_aware_retrieval()
    rag_with_reranking_comparison()
    rag_with_sources_and_metadata()
    print("\nAll RAG examples completed!")
