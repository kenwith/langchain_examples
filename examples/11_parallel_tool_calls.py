"""Example 11: Parallel Tool Calls.

This example demonstrates how to use `bind_tools` and `ToolNode` to execute
multiple tool calls in parallel.

| Example | Description |
|---------|-------------|
| 11 | Parallel tool calls with `bind_tools` and `ToolNode`. |
"""

import os
import warnings

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing import Annotated, TypedDict


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


class State(TypedDict):
    messages: Annotated[list, add_messages]


def create_graph():
    """Create a LangGraph agent that can execute parallel tool calls."""
    # Provider-agnostic model initialization.
    # Set CHAT_MODEL and CHAT_MODEL_PROVIDER to use a different model/provider.
    model = init_chat_model(
        model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
        model_provider=os.getenv("CHAT_MODEL_PROVIDER", "openai"),
        temperature=0,
    )

    # Detect whether the model supports parallel tool calls.
    parallel_supported = getattr(model, "parallel_tool_calls", True)
    if not parallel_supported:
        warnings.warn(
            "The selected model does not support parallel tool calls. "
            "Falling back to sequential execution."
        )

    tools = [add, multiply]
    tool_node = ToolNode(tools, parallel=parallel_supported)
    model_with_tools = model.bind_tools(tools)

    def call_model(state):
        return {"messages": [model_with_tools.invoke(state["messages"])]}

    def should_continue(state):
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tools"
        return END

    graph = StateGraph(State)
    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "agent")

    return graph.compile()


def main():
    """Run a simple demo of parallel tool calls."""
    app = create_graph()

    # This prompt should trigger two independent tool calls in one response.
    result = app.invoke(
        {"messages": [("human", "What is 2 + 3 and 4 * 5?")]}
    )

    for message in result["messages"]:
        if hasattr(message, "pretty_print"):
            message.pretty_print()
        else:
            print(message)


if __name__ == "__main__":
    main()
