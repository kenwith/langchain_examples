"""# 66. Ollama Async Agentic RAG

| Field | Value |
|-------|-------|
| Description | An agentic RAG flow using async retrieval, tool-calling agent, and Ollama. |
| Tags | async, rag, agent, ollama, tool-calling |
| Model | llama3.1 (via Ollama) |
| Embeddings | nomic-embed-text (via Ollama) |
| Vector Store | InMemoryVectorStore |
| Agent Type | Tool-calling agent |
| Example | examples/66_ollama_async_agentic_rag.py |
"""

import asyncio
import os

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings


def load_documents() -> list[Document]:
    """Load sample documents for the RAG example."""
    content = [
        "LangChain is a framework for developing applications powered by language models.",
        "Agents in LangChain use tools to interact with external systems and make decisions.",
        "Ollama allows running large language models locally on a machine.",
        "Asynchronous retrieval enables non-blocking document search in RAG pipelines.",
        "Tool-calling agents can decide when to invoke external tools during generation.",
    ]
    return [Document(page_content=text) for text in content]


def create_retriever_tool(vectorstore: InMemoryVectorStore):
    """Create an async retriever tool backed by the vector store."""

    @tool
    async def retrieve_documents(query: str) -> str:
        """Retrieve relevant documents for the given query."""
        docs = await vectorstore.asimilarity_search(query, k=3)
        return "\n\n".join(doc.page_content for doc in docs)

    return retrieve_documents


def create_agent(model, tools):
    """Create a tool-calling agent with the given model and tools."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Use the retrieve_documents tool to answer questions.",
            ),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    agent = create_tool_calling_agent(model, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


async def main() -> None:
    """Run the async agentic RAG example."""
    documents = load_documents()

    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1")
    ollama_embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

    embeddings = OllamaEmbeddings(
        model=ollama_embedding_model,
        base_url=ollama_base_url,
    )

    vectorstore = InMemoryVectorStore(embeddings)
    await vectorstore.aadd_documents(documents)

    retriever_tool = create_retriever_tool(vectorstore)

    model = init_chat_model(
        model=ollama_model,
        model_provider="ollama",
        base_url=ollama_base_url,
        temperature=0,
    )

    agent = create_agent(model, [retriever_tool])

    result = await agent.ainvoke(
        {"input": "What is LangChain and how does it use agents?"}
    )
    print(result["output"])


if __name__ == "__main__":
    asyncio.run(main())
