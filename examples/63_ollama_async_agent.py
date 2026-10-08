"""
Example 63: Ollama Async Agent

| # | Example | Description |
|---|---------|-------------|
| 63 | ollama_async_agent | An async agent loop with tool calling using Ollama. |

This example demonstrates an asynchronous agent loop that uses tool calling with
Ollama. The chat model is initialized using the provider-agnostic
`init_chat_model` function.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| OLLAMA_MODEL | The Ollama model name | llama3.1 |
| OLLAMA_BASE_URL | The Ollama server base URL | http://localhost:11434 |
"""

import asyncio
import os

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool


@tool
def add(a: int, b: int) -> int:
    """Add two integers and return the result."""
    return a + b


async def run_agent(query: str, max_iterations: int = 5) -> str:
    """Run the async agent loop with tool calling."""
    llm = init_chat_model(
        model=os.getenv("OLLAMA_MODEL", "llama3.1"),
        model_provider="ollama",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    bound_llm = llm.bind_tools([add])

    messages = [
        SystemMessage(content="You are a helpful assistant. Use the provided tools when needed."),
        HumanMessage(content=query),
    ]

    for _ in range(max_iterations):
        response = await bound_llm.ainvoke(messages)
        messages.append(response)

        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_result = add.invoke(tool_call.args)
                messages.append(
                    ToolMessage(content=str(tool_result), tool_call_id=tool_call.id)
                )
        else:
            return response.content

    return "Max iterations reached without a final answer."


async def main() -> None:
    """Run the example agent with a sample query."""
    answer = await run_agent("What is 2 + 3?")
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
