"""
Example 65: Async Agent with Memory using Ollama and LangGraph

This example demonstrates how to build an asynchronous agent with conversation memory
using Ollama as the chat model provider and LangGraph for orchestration.
The agent uses a simple arithmetic tool and maintains conversation history across turns.

The example exposes an `async def run_agent` helper that invokes the agent for a single
turn with a given message and thread ID. This helper can be reused in tests or other
applications to exercise the agent's memory and tool-calling behavior.
"""

import asyncio
import os
from typing import TypedDict

from langchain.chat_models.base import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


class AgentState(TypedDict):
    messages: list


def create_agent():
    """Create the async LangGraph agent with memory."""
    # Initialize the chat model with Ollama
    model = init_chat_model(
        model=os.getenv("OLLAMA_MODEL", "llama3.1"),
        provider="ollama",
        temperature=0,
    )

    # Bind the tool to the model
    tools = [add]
    model_with_tools = model.bind_tools(tools)

    # Define the model node
    async def model_node(state: AgentState):
        messages = state["messages"]
        response = await model_with_tools.ainvoke(messages)
        return {"messages": [response]}

    # Define the tool node
    tool_node = ToolNode(tools)

    # Define the conditional edge function
    def should_continue(state: AgentState):
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tools"
        return END

    # Build the graph
    graph = StateGraph(AgentState)
    graph.add_node("model", model_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "model")

    # Compile with a checkpointer for memory
    checkpointer = InMemorySaver()
    return graph.compile(checkpointer=checkpointer)


async def run_agent(agent, message: str, thread_id: str = "1") -> str:
    """Run a single turn against the agent and return the response text."""
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=message)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    return result["messages"][-1].content


async def run_conversation():
    """Run a two-turn conversation demonstrating memory."""
    agent = create_agent()

    # First turn
    first_response = await run_agent(agent, "What is 2 + 3?")
    print(first_response)

    # Second turn, referencing previous context
    second_response = await run_agent(agent, "Now add 4 to that result.")
    print(second_response)


if __name__ == "__main__":
    asyncio.run(run_conversation())
