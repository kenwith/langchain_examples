"""# Async Parallel Tool Calls

## Overview

This example combines `asyncio.gather` with LangChain tool calling to execute
multiple tool invocations concurrently in an async workflow.

## Tools

Two async tools are defined: `get_weather` and `get_time`.

## Workflow

1. Bind the tools to a provider-agnostic chat model via `init_chat_model`.
2. Ask the model a question that should trigger multiple tool calls.
3. Run all requested tool calls in parallel with `asyncio.gather`.
4. Print the combined tool results.

## Environment

Set the following environment variables:

- `MODEL` – the language model name (e.g., `gpt-4o`, `claude-3-5-sonnet`)
- `MODEL_PROVIDER` – the provider name (e.g., `openai`, `anthropic`)
- Provider-specific API keys (e.g., `OPENAI_API_KEY`).

Use `os.getenv` in your own credentials; never hard-code secrets.
"""

import asyncio
import os

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool


@tool
async def get_weather(city: str) -> str:
    """Return the current weather for a given city.

    Args:
        city: The city name.

    Returns:
        A short weather report.
    """
    await asyncio.sleep(1)
    return f"Weather in {city}: 22°C and sunny"


@tool
async def get_time(city: str) -> str:
    """Return the current time for a given city.

    Args:
        city: The city name.

    Returns:
        A short time report.
    """
    await asyncio.sleep(1)
    return f"Time in {city}: 12:00 PM"


async def run_parallel_tool_calls() -> str:
    """Invoke the model and run its tool calls in parallel."""
    model = init_chat_model(
        os.getenv("MODEL"),
        model_provider=os.getenv("MODEL_PROVIDER"),
    )

    model_with_tools = model.bind_tools([get_weather, get_time])

    response = await model_with_tools.ainvoke(
        [
            (
                "user",
                "What's the weather in Paris and Tokyo? Also, what time is it in London?",
            )
        ]
    )

    if not response.tool_calls:
        return response.content

    tool_map = {
        "get_weather": get_weather,
        "get_time": get_time,
    }

    tasks = [
        tool_map[tool_call["name"]].ainvoke(tool_call["args"])
        for tool_call in response.tool_calls
    ]

    results = await asyncio.gather(*tasks)

    return "\n".join(str(result) for result in results)


if __name__ == "__main__":
    output = asyncio.run(run_parallel_tool_calls())
    print(output)
