"""Example 43: Build a Tool-Calling Agent with Ollama.

This example demonstrates how to create a tool-calling agent using Ollama
with the provider-agnostic `init_chat_model` function. The agent uses
Ollama's tool-calling support to decide when to invoke arithmetic tools.

Setup:
    Install Ollama from https://ollama.com and pull a tool-capable model:
        ollama pull llama3.1

Run:
    python examples/43_ollama_agent.py
"""

import os

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool


# ## Tools
@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


# ## Agent
def build_agent():
    """Build a tool-calling agent with Ollama."""
    model = os.getenv("OLLAMA_MODEL", "llama3.1")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    llm = init_chat_model(
        model=model,
        provider="ollama",
        base_url=base_url,
        temperature=0,
    )

    tools = [add, multiply]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Use tools when needed."),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


# ## Main
def main():
    """Run the agent with a sample query."""
    agent = build_agent()
    response = agent.invoke({"input": "What is 17 * 23? Use the multiply tool."})
    print(response["output"])


if __name__ == "__main__":
    main()
