"""Example 53: Ollama Agent with Conversation Memory.

This example demonstrates how to build an agent with conversation memory
using Ollama as the model provider. The agent uses a simple weather tool
and remembers context across turns with an in-memory checkpointer.

Dependencies:
    pip install langchain langchain-ollama langgraph
"""

# README-table style header
# Example: 53. Ollama Agent with Memory
# Provider: Ollama
# Model: llama3.2 (default, configurable via OLLAMA_MODEL)
# Tools: get_weather
# Memory: InMemorySaver

import os

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver


@tool
def get_weather(city: str) -> str:
    """Return the current weather for a given city."""
    # This is a mock tool; replace with a real weather API.
    return f"The weather in {city} is sunny with a high of 75°F."


def build_agent():
    """Create a LangChain agent with an Ollama model and conversation memory."""
    model_name = os.getenv("OLLAMA_MODEL", "llama3.2")
    model = init_chat_model(model=f"ollama:{model_name}", temperature=0)

    checkpointer = InMemorySaver()

    agent = create_agent(
        model=model,
        tools=[get_weather],
        system_prompt="You are a helpful assistant with access to a weather tool.",
        checkpointer=checkpointer,
    )
    return agent


def main():
    """Run a short conversation to demonstrate memory."""
    agent = build_agent()

    config = {"configurable": {"thread_id": "demo-thread"}}

    # First turn: ask about weather
    response = agent.invoke(
        {"messages": [("human", "What's the weather in San Francisco?")]},
        config,
    )
    print("User: What's the weather in San Francisco?")
    print("Assistant:", response["messages"][-1].content)
    print()

    # Follow-up question that requires remembering the previous context.
    response = agent.invoke(
        {"messages": [("human", "What was the city I just asked about?")]},
        config,
    )
    print("User: What was the city I just asked about?")
    print("Assistant:", response["messages"][-1].content)


if __name__ == "__main__":
    main()
