"""Example 92: Ollama RAG Fusion.

This example demonstrates a provider-agnostic RAG-Fusion chain using Ollama for
chat, embeddings, and multi-query retrieval. The chain generates multiple query
variations, retrieves relevant documents for each, fuses the results with
reciprocal rank fusion, and produces a final answer.

| Component          | Model/Setting       |
|--------------------|---------------------|
| Chat model         | llama3.2 (Ollama)   |
| Embedding model    | nomic-embed-text    |
| Vector store       | InMemoryVectorStore |

Run the script to execute the demo question and print the answer to stdout.
"""

import os
from typing import List

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.output_parsers import CommaSeparatedListOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")


def load_documents() -> List[Document]:
    """Return a small set of documents for the RAG-Fusion demo."""
    return [
        Document(
            page_content="Ollama is a tool for running large language models locally. "
            "It supports models such as LLaMA and Mistral and integrates with "
            "LangChain through the Ollama provider."
        ),
        Document(
            page_content="Retrieval-Augmented Generation (RAG) combines a language "
            "model with external knowledge sources. Relevant documents are retrieved "
            "and included in the prompt to improve factual accuracy and reduce "
            "hallucinations."
        ),
        Document(
            page_content="Reciprocal Rank Fusion (RRF) combines multiple ranked lists "
            "by summing a score of 1 divided by a rank offset. This method is robust "
            "and does not rely on score normalization across different retrievers."
        ),
        Document(
            page_content="RAG-Fusion extends RAG by generating multiple query "
            "variations, retrieving documents for each variation, and fusing the "
            "results with Reciprocal Rank Fusion. This approach can surface more "
            "relevant information than a single query."
        ),
    ]


def create_vector_store(documents: List[Document]) -> InMemoryVectorStore:
    """Split documents and store embeddings in an in-memory vector store."""
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = text_splitter.split_documents(documents)
    return InMemoryVectorStore.from_documents(chunks, embedding=embeddings)


def build_query_variation_chain(chat_model):
    """Build a chain that generates multiple search query variations."""
    parser = CommaSeparatedListOutputParser()
    prompt = ChatPromptTemplate.from_template(
        "Generate {num_queries} distinct search queries related to the original "
        "query. Use different wording, perspectives, and levels of specificity.\n"
        "Original query: {question}\n"
        "{format_instructions}"
    ).partial(format_instructions=parser.get_format_instructions())
    return prompt | chat_model | parser


def reciprocal_rank_fusion(
    retrieved_documents: List[List[Document]], k: int = 60
) -> List[Document]:
    """Fuse multiple retrieved document lists using reciprocal rank fusion."""
    fused_scores = {}
    for docs in retrieved_documents:
        for rank, doc in enumerate(docs):
            content = doc.page_content
            if content not in fused_scores:
                fused_scores[content] = 0.0
            fused_scores[content] += 1.0 / (rank + k)
    ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [Document(page_content=content) for content, _ in ranked]


def retrieve_fused_documents(
    question: str,
    vector_store: InMemoryVectorStore,
    query_chain,
    num_queries: int = 4,
    k: int = 4,
) -> List[Document]:
    """Generate query variations, retrieve for each, and fuse results."""
    query_variations = query_chain.invoke(
        {"question": question, "num_queries": num_queries}
    )
    all_retrieved = [
        vector_store.similarity_search(query, k=k) for query in query_variations
    ]
    return reciprocal_rank_fusion(all_retrieved)


def answer_question(
    question: str,
    chat_model,
    vector_store: InMemoryVectorStore,
    query_chain,
    num_queries: int = 4,
    k: int = 4,
) -> str:
    """Answer a question using the RAG-Fusion strategy."""
    fused_docs = retrieve_fused_documents(
        question, vector_store, query_chain, num_queries, k
    )
    context = "\n\n".join(doc.page_content for doc in fused_docs[:5])
    prompt = ChatPromptTemplate.from_template(
        "Answer the question based solely on the provided context.\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}\n\n"
        "Answer:"
    )
    chain = prompt | chat_model | StrOutputParser()
    return chain.invoke({"context": context, "question": question})


def build_chain():
    """Create the vector store, chat model, and query variation chain."""
    documents = load_documents()
    vector_store = create_vector_store(documents)
    chat_model = init_chat_model(
        CHAT_MODEL,
        model_provider="ollama",
        temperature=0.1,
    )
    query_chain = build_query_variation_chain(chat_model)
    return chat_model, vector_store, query_chain


if __name__ == "__main__":
    demo_question = "What is reciprocal rank fusion and why is it useful for RAG?"
    chat_model, vector_store, query_chain = build_chain()
    answer = answer_question(demo_question, chat_model, vector_store, query_chain)
    print(answer)
