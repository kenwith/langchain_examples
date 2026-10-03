"""50. Contextual Compression.

Demonstrates how to use ContextualCompressionRetriever with an LLMChainExtractor
to compress retrieved documents to only the most relevant parts.

How the technique works:
- A base retriever first fetches candidate documents for a query.
- A document compressor then processes each document with an LLM, extracting
  only the sentences that are relevant to the query.
- The result is a shorter, more focused set of documents that reduces noise and
  saves tokens when feeding context to a downstream LLM.

This example is provider-agnostic: it uses init_chat_model() to create a chat
model based on environment variables.

How to run:
1. Install LangChain dependencies (langchain, langchain-core, and the package
   for your chosen model provider).
2. Set the API key for your provider, e.g.:
   export OPENAI_API_KEY="your-key"
   or:
   export ANTHROPIC_API_KEY="your-key"
3. Optionally override the model and provider:
   export MODEL_NAME="gpt-4o"
   export MODEL_PROVIDER="openai"
   or:
   export MODEL_NAME="claude-3-5-sonnet-20241022"
   export MODEL_PROVIDER="anthropic"
4. Run the example:
   python examples/50_contextual_compression.py
"""

import os
from typing import List

from langchain.chat_models import init_chat_model
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class DummyRetriever(BaseRetriever):
    """A simple retriever that returns a fixed set of documents."""

    documents: List[Document]

    def _get_relevant_documents(self, query: str) -> List[Document]:
        return self.documents


def create_dummy_retriever() -> DummyRetriever:
    """Create a retriever with a few sample documents."""
    documents = [
        Document(
            page_content=(
                "LangChain is a framework for developing applications powered by "
                "large language models. It provides standard interfaces for chains, "
                "agents, and retrieval-augmented generation."
            ),
            metadata={"source": "langchain_intro"},
        ),
        Document(
            page_content=(
                "Contextual compression is a technique in LangChain that reduces the "
                "amount of irrelevant information in retrieved documents. It uses an "
                "LLM to extract only the parts that are relevant to a query."
            ),
            metadata={"source": "langchain_compression"},
        ),
        Document(
            page_content=(
                "The LLMChainExtractor uses a language model to identify and extract "
                "the most relevant sentences from a document for a given query. It "
                "returns a new document with only the extracted content."
            ),
            metadata={"source": "llm_chain_extractor"},
        ),
    ]
    return DummyRetriever(documents=documents)


def build_compression_retriever():
    """Build a ContextualCompressionRetriever with an LLMChainExtractor."""
    # Create a provider-agnostic chat model. The model and provider are read
    # from environment variables. For example:
    #   OPENAI_API_KEY=... MODEL_NAME=gpt-4o MODEL_PROVIDER=openai
    # or:
    #   ANTHROPIC_API_KEY=... MODEL_NAME=claude-3-5-sonnet-20241022 MODEL_PROVIDER=anthropic
    llm = init_chat_model(
        model=os.getenv("MODEL_NAME", "gpt-4o"),
        model_provider=os.getenv("MODEL_PROVIDER", "openai"),
        temperature=0,
    )

    compressor = LLMChainExtractor.from_llm(llm)
    base_retriever = create_dummy_retriever()

    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )


def main():
    """Run the contextual compression example."""
    retriever = build_compression_retriever()
    query = "What is contextual compression?"
    compressed_docs = retriever.invoke(query)

    print(f"Query: {query}\n")
    print("Compressed documents:")
    for i, doc in enumerate(compressed_docs, 1):
        print(f"\n--- Document {i} ---")
        print(doc.page_content)
        if doc.metadata:
            print(f"Metadata: {doc.metadata}")


if __name__ == "__main__":
    main()
