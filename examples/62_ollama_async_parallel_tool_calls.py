"""Example: Async Parallel Tool Calls with Ollama.

This example demonstrates how to combine asynchronous execution with parallel tool
calls using Ollama and a local chat model. The model is initialized in a
provider-agnostic way with `init_chat_model`, and multiple queries are processed
concurrently with `asyncio.gather`. Tool calls from a single model response are
executed in parallel as well.

| Component | Setting |
|-----------|---------|
| Model | llama3.1 (or `OLLAMA_MODEL`) |
| Provider | ollama |
| Base URL | http://localhost:11434 (or `OLLAMA_BASE_URL`) |
"""

import asyncio
import os

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


@tool
def get_capital(country: str) -> str:
    """Return the capital of a country."""
    capitals = {
        "france": "Paris",
        "japan": "Tokyo",
        "germany": "Berlin",
        "italy": "Rome",
    }
    return capitals.get(country.lower(), "Unknown")


async def execute_tool_call(tool_call: dict, tools_by_name: dict) -> ToolMessage:
    """Execute a single tool call and return a ToolMessage."""
    tool_fn = tools_by_name[tool_call["name"]]
    result = await asyncio.to_thread(tool_fn.invoke, tool_call["args"])
    return ToolMessage(content=str(result), tool_call_id=tool_call["id"])


async def run_query(model, query: str, tools_by_name: dict) -> str:
    """Run a query with tool calling and return the final answer."""
    messages = [HumanMessage(content=query)]
    response = await model.bind_tools(list(tools_by_name.values())).ainvoke(messages)

    if response.tool_calls:
        tool_messages = await asyncio.gather(
            *(execute_tool_call(tc, tools_by_name) for tc in response.tool_calls)
        )
        messages.extend([response, *tool_messages])
        final_response = await model.ainvoke(messages)
        return final_response.content

    return response.content


async def main() -> None:
    """Run multiple queries concurrently with parallel tool calls."""
    model = init_chat_model(
        model=os.getenv("OLLAMA_MODEL", "llama3.1"),
        model_provider="ollama",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0,
    )

    tools = [add, multiply, get_capital]
    tools_by_name = {tool.name: tool for tool in tools}

    queries = [
        "What is 17 * 3 and 23 + 45?",
        "What are the capitals of France and Japan?",
        "Calculate 12 * 8 and 99 + 1.",
    ]

    results = await asyncio.gather(
        *(run_query(model, query, tools_by_name) for query in queries)
    )

    for query, result in zip(queries, results):
        print(f"Query: {query}\nAnswer: {result}\n")


if __name__ == "__main__":
    asyncio.run(main())
