"""Example 43: Build a Tool-Calling Agent with Ollama.

This example demonstrates how to create a tool-calling agent using Ollama
with the provider-agnostic `init_chat_model` function. The agent uses
Ollama's tool-calling support to decide when to invoke arithmetic tools.

How the agent selects tools:
    The `create_tool_calling_agent` function registers the provided tools
    (add and multiply) with the model. The model receives the user's request
    and the tool schemas, then decides whether to call a tool based on the
    conversation. If it chooses to call a tool, the `AgentExecutor` invokes
    the tool and feeds the result back to the model.

Setup:
    Install Ollama from https://ollama.com and pull a tool-capable model:
        ollama pull llama3.1

Environment variables:
    OLLAMA_MODEL: model name (default: llama3.1)
    OLLAMA_BASE_URL: Ollama server URL (default: http://localhost:11434)

Run:
    python examples/43_ollama_agent.py [query]
"""

import os
import sys

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
def main(query: str = "What is 17 * 23? Use the multiply tool."):
    """Run the agent with a sample query."""
    agent = build_agent()
    response = agent.invoke({"input": query})
    print(response["output"])


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "What is 17 * 23? Use the multiply tool."
    main(query)
