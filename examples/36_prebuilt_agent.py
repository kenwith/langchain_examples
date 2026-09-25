"""Example: Prebuilt Agent with Tool Calling.

| | |
|---|---|
| Name | 36_prebuilt_agent |
| Description | Demonstrates using `create_agent` from `langgraph.prebuilt` to create a simple tool-calling agent. |
| Dependencies | langgraph, langchain, and a chat model provider (e.g., langchain-openai) |

This example creates a small agent that can call arithmetic tools. It uses
`init_chat_model` to create a model in a provider-agnostic way and then wraps
it with `create_agent` for tool calling.

Run:
    python 36_prebuilt_agent.py
"""

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.prebuilt import create_agent

import os


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def get_model():
    """Create a chat model using provider-agnostic init_chat_model."""
    return init_chat_model(
        model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        model_provider=os.getenv("MODEL_PROVIDER", "openai"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def build_agent():
    """Build a prebuilt agent with the arithmetic tools."""
    model = get_model()
    tools = [multiply, add]
    return create_agent(model=model, tools=tools)


def main():
    """Run a simple query through the prebuilt agent."""
    agent = build_agent()
    result = agent.invoke({"messages": [("user", "What is 3 times 4?")]})
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
