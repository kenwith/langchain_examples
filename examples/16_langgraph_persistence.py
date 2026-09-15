"""Example 16: LangGraph Persistence with MemorySaver.

This example demonstrates how to use LangGraph's MemorySaver checkpointer to
persist state across multiple invocations of the same compiled graph.
"""

# ---------------------------------------------------------------------------
# Example 16: LangGraph Persistence with MemorySaver
# Description: Persist state across multiple LangGraph invocations using
#              MemorySaver checkpointing.
# Run: python examples/16_langgraph_persistence.py
# Environment variables:
#   MODEL            (optional) Chat model name (default: gpt-4o-mini)
#   MODEL_PROVIDER   (optional) Model provider (default: openai)
# ---------------------------------------------------------------------------

from __future__ import annotations

import os
from typing import Annotated, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


class State(TypedDict):
    """State of the conversation graph."""
    messages: Annotated[list, add_messages]


def chat_node(state: State, llm) -> dict:
    """Call the chat model with the full conversation history."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def build_graph(llm, checkpointer):
    """Build and compile a single-node LangGraph with persistence."""
    graph = StateGraph(State)
    graph.add_node("chat", lambda state: chat_node(state, llm))
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)
    return graph.compile(checkpointer=checkpointer)


def main() -> None:
    """Run the persistence demo with two thread IDs."""
    llm = init_chat_model(
        os.getenv("MODEL", "gpt-4o-mini"),
        model_provider=os.getenv("MODEL_PROVIDER", "openai"),
    )

    memory = MemorySaver()
    app = build_graph(llm, memory)

    thread_a = {"configurable": {"thread_id": "thread-a"}}
    thread_b = {"configurable": {"thread_id": "thread-b"}}

    print("--- Invocation 1: thread-a ---")
    result = app.invoke(
        {"messages": [HumanMessage(content="Hello, my name is Ada.")]},
        config=thread_a,
    )
    print(result["messages"][-1].content)

    print("\n--- Invocation 2: thread-a ---")
    result = app.invoke(
        {"messages": [HumanMessage(content="What is my name?")]},
        config=thread_a,
    )
    print(result["messages"][-1].content)

    print("\n--- Invocation 3: thread-b ---")
    result = app.invoke(
        {"messages": [HumanMessage(content="What is my name?")]},
        config=thread_b,
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
