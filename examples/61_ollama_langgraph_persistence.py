"""Example 61: Persist LangGraph agent state across runs with Ollama.

This example builds a small tool-calling agent with LangGraph and an
Ollama-hosted chat model loaded through
init_chat_model(..., model_provider="ollama"). A MemorySaver
checkpointer persists the full conversation state, allowing the agent
to retain context across separate invocations when the same thread_id
is reused.
"""

import os
from typing import Annotated

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict


@tool
def get_weather(city: str) -> str:
    """Return a mock weather report for a given city."""
    return f"Weather in {city}: sunny, 22°C"


class AgentState(TypedDict):
    """State for the agent graph with automatic message accumulation."""
    messages: Annotated[list, add_messages]


def build_agent():
    """Build and compile a tool-calling agent with a MemorySaver checkpointer."""
    model = init_chat_model(
        os.getenv("OLLAMA_MODEL", "llama3.1"),
        model_provider="ollama",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0,
    ).bind_tools([get_weather])

    def call_model(state: AgentState):
        return {"messages": [model.invoke(state["messages"])]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode([get_weather]))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=MemorySaver())


def main():
    """Run the agent across multiple invocations to demonstrate persistence."""
    agent = build_agent()

    thread_a = {"configurable": {"thread_id": "thread-a"}}
    thread_b = {"configurable": {"thread_id": "thread-b"}}

    first = agent.invoke(
        {"messages": [("human", "Hi, I am Alice. What is the weather in Berlin?")]},
        thread_a,
    )
    print("First run (thread-a):")
    print(first["messages"][-1].content)
    print()

    second = agent.invoke(
        {"messages": [("human", "What is my name?")]},
        thread_a,
    )
    print("Second run (thread-a):")
    print(second["messages"][-1].content)
    print()

    third = agent.invoke(
        {"messages": [("human", "What is my name?")]},
        thread_b,
    )
    print("Third run (thread-b):")
    print(third["messages"][-1].content)


if __name__ == "__main__":
    main()
