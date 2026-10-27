"""Example: 80. Ollama Async LangGraph Agentic RAG

This example demonstrates an agentic Retrieval-Augmented Generation (RAG)
workflow using Ollama for local inference, LangGraph for agentic control,
and asynchronous execution.
"""

import asyncio
import os

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langgraph.prebuilt import create_react_agent

# | Field      | Value                          |
# |------------|--------------------------------|
# | Example    | 80 Ollama Async LangGraph      |
# |            | Agentic RAG                    |
# | Requires   | Ollama running locally         |
# | Model      | llama3.1 (or OLLAMA_CHAT_MODEL)|
# | Embeddings | nomic-embed-text               |

# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the search_documents tool to look up "
    "facts before answering. Base your answer only on the retrieved content."
)

# ---------------------------------------------------------------------------
# Retrieval setup
# ---------------------------------------------------------------------------


def build_retriever():
    """Build an in-memory vector store retriever from sample documents."""
    texts = [
        "Project Corail is a universal orchestrator designed to unify human and machine workflows.",
        "Project Corail was announced in 2025 and the first stable release is scheduled for July 2026.",
        "The LangGraph framework provides low-level primitives for building agentic applications.",
        "Ollama enables local large language model inference with a simple REST API.",
    ]
    embeddings = OllamaEmbeddings(
        model=OLLAMA_EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    vectorstore = InMemoryVectorStore.from_texts(texts, embedding=embeddings)
    return vectorstore.as_retriever()


def create_search_tool(retriever):
    """Create a search tool that uses the provided retriever."""

    @tool
    async def search_documents(query: str) -> str:
        """Search the knowledge base for relevant documents. Use this tool when the answer may not be in your training data."""
        docs = await retriever.ainvoke(query)
        return "\n\n".join(doc.page_content for doc in docs)

    return search_documents


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------


async def main() -> None:
    """Run the agentic RAG demo."""
    retriever = build_retriever()
    search_tool = create_search_tool(retriever)

    model = init_chat_model(
        model=OLLAMA_CHAT_MODEL,
        model_provider="ollama",
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )

    agent = create_react_agent(
        model,
        [search_tool],
        state_modifier=SYSTEM_PROMPT,
    )

    question = "What is Project Corail and when was it announced?"
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": question}]}
    )

    for message in result["messages"]:
        if message.type == "ai":
            print(message.content)


if __name__ == "__main__":
    asyncio.run(main())
