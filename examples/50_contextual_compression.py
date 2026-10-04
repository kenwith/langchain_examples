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

To adapt this example:
- Pass your own `llm` or `base_retriever` to `build_compression_retriever()`
  to use custom models or retrieval logic without modifying the helper.
"""

import os
from typing import List, Optional

from langchain.chat_models import init_chat_model
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
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


def build_compression_retriever(
    llm: Optional[BaseChatModel] = None,
    base_retriever: Optional[BaseRetriever] = None,
) -> ContextualCompressionRetriever:
    """Build a ContextualCompressionRetriever with an LLMChainExtractor.

    Args:
        llm: A chat model instance to use for compression. If None, a model is
            created from the MODEL_NAME and MODEL_PROVIDER environment variables.
        base_retriever: A retriever that returns candidate documents. If None,
            a DummyRetriever with sample documents is used.

    Returns:
        A configured ContextualCompressionRetriever.
    """
    if llm is None:
        llm = init_chat_model(
            model=os.getenv("MODEL_NAME", "gpt-4o"),
            model_provider=os.getenv("MODEL_PROVIDER", "openai"),
            temperature=0,
        )

    if base_retriever is None:
        base_retriever = create_dummy_retriever()

    compressor = LLMChainExtractor.from_llm(llm)

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
