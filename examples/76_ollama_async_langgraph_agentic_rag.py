"""
Example: Async LangGraph Agentic RAG with Ollama

This example demonstrates an asynchronous agentic RAG workflow using LangGraph
and a local Ollama model. The agent decides whether to retrieve relevant
documents or answer directly, then produces a final answer grounded in the
retrieved context.

| Provider | Model       | Use case        |
|----------|-------------|-----------------|
| Ollama   | llama3.1    | Agentic RAG     |

Requirements:
- Ollama running locally with a model that supports tool calling (e.g., llama3.1)
- langchain, langgraph, langchain-community, langchain-ollama, faiss-cpu

Run:
    python examples/76_ollama_async_langgraph_agentic_rag.py
"""

import asyncio
import os

from langchain.chat_models import init_chat_model
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import StateGraph as StateGraphType
from typing import TypedDict, List, Any


# -----------------------------------------------------------------------------
# State definition
# -----------------------------------------------------------------------------

class AgentState(TypedDict):
    """State passed between graph nodes."""
    question: str
    messages: List[Any]          # chat message history
    context: List[Document]      # retrieved documents
    answer: str                  # final answer


# -----------------------------------------------------------------------------
# Tools and vector store setup
# -----------------------------------------------------------------------------

@tool
def retrieve_documents(query: str) -> str:
    """Retrieve relevant documents for the given query."""
    # In a real application, this would query a persistent vector store.
    # Here we use an in-memory FAISS index built from sample documents.
    docs = [
        Document(
            page_content="LangGraph is a library for building stateful, "
                        "agentic applications with LLMs. It provides "
                        "fine-grained control over the flow of data and "
                        "state through a graph of nodes and edges.",
            metadata={"source": "langgraph-docs"}
        ),
        Document(
            page_content="Ollama allows running large language models locally "
                        "with a simple API. It supports models like llama3.1 "
                        "and mistral, and can be used with LangChain via "
                        "langchain-ollama.",
            metadata={"source": "ollama-docs"}
        ),
        Document(
            page_content="Agentic RAG combines retrieval-augmented generation "
                        "with an agent loop, enabling the model to decide "
                        "when to retrieve and how to synthesize answers.",
            metadata={"source": "rag-guide"}
        ),
    ]

    # Build a simple vector store with Ollama embeddings.
    embeddings = OllamaEmbeddings(model="llama3.1")
    vector_store = FAISS.from_documents(docs, embeddings)

    # Retrieve top-k relevant documents.
    retrieved = vector_store.similarity_search(query, k=2)
    return "\n\n".join(doc.page_content for doc in retrieved)


# -----------------------------------------------------------------------------
# Graph nodes
# -----------------------------------------------------------------------------

async def agent_node(state: AgentState) -> dict:
    """
    The agent decides whether to retrieve or answer directly.
    It uses the LLM with the retrieval tool and returns the model's response.
    """
    llm = init_chat_model(
        model="llama3.1",
        provider="ollama",
        temperature=0.1,
    )
    llm_with_tools = llm.bind_tools([retrieve_documents])

    system_prompt = (
        "You are a helpful assistant. If you need external information to "
        "answer the question, use the retrieve_documents tool. Otherwise, "
        "answer directly. Always be concise and accurate."
    )

    # Build the message list for the model.
    from langchain_core.messages import HumanMessage, SystemMessage
    messages = [SystemMessage(content=system_prompt)]
    messages.append(HumanMessage(content=state["question"]))

    response = await llm_with_tools.ainvoke(messages)

    # Return the new messages and the response itself.
    return {"messages": [response]}


async def retrieve_node(state: AgentState) -> dict:
    """
    Execute the retrieval tool call and store the context.
    """
    # Find the last tool call in the messages.
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls

    if not tool_calls:
        raise ValueError("No tool call found in the last message.")

    # Execute the first tool call (retrieve_documents).
    tool_call = tool_calls[0]
    query = tool_call["args"]["query"]
    context_text = retrieve_documents.invoke(query)

    # Convert the string back to a Document for state.
    context = [Document(page_content=context_text)]

    return {"context": context}


async def answer_node(state: AgentState) -> dict:
    """
    Generate the final answer using the retrieved context (if any).
    """
    llm = init_chat_model(
        model="llama3.1",
        provider="ollama",
        temperature=0.2,
    )

    # Build a prompt that includes the context.
    context_text = "\n\n".join(doc.page_content for doc in state.get("context", []))
    if context_text:
        prompt = (
            f"Use the following context to answer the question.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {state['question']}\n"
            f"Answer:"
        )
    else:
        prompt = (
            f"Answer the following question directly.\n\n"
            f"Question: {state['question']}\n"
            f"Answer:"
        )

    from langchain_core.messages import HumanMessage
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return {"answer": response.content}


# -----------------------------------------------------------------------------
# Graph construction
# -----------------------------------------------------------------------------

def build_graph() -> StateGraphType:
    """
    Build the async LangGraph for agentic RAG.
    """
    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer", answer_node)

    graph.add_edge(START, "agent")

    # Conditional edge: if the agent issued a tool call, go to retrieve;
    # otherwise, go directly to answer.
    def should_retrieve(state: AgentState) -> str:
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "retrieve"
        return "answer"

    graph.add_conditional_edges(
        "agent",
        should_retrieve,
        {"retrieve": "retrieve", "answer": "answer"},
    )

    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)

    return graph


# -----------------------------------------------------------------------------
# Main demo
# -----------------------------------------------------------------------------

async def main() -> None:
    """Run a sample query through the agentic RAG graph."""
    # Ensure Ollama is running and the model is available.
    if not os.getenv("OLLAMA_BASE_URL"):
        os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"

    graph = build_graph()
    app = graph.compile()

    question = "What is LangGraph and how does it relate to agentic RAG?"
    initial_state = {
        "question": question,
        "messages": [],
        "context": [],
        "answer": "",
    }

    result = await app.ainvoke(initial_state)
    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")


if __name__ == "__main__":
    asyncio.run(main())
