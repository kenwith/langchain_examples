"""Example 94: Ollama Async Streaming with Tools.

.. list-table::
   :header-rows: 1

   * - Requirement
     - Method
   * - Model
     - Ollama (e.g., llama3.1)
   * - Provider
     - ollama
   * - Streaming
     - Yes
   * - Tools
     - Yes
   * - Async
     - Yes

This example demonstrates how to stream responses from an Ollama model
while using tools asynchronously. The model is created with
``init_chat_model``, making the example provider-agnostic.
"""

from __future__ import annotations

import asyncio
import os

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool


@tool
def get_current_weather(location: str) -> str:
    """Get the current weather for a location.

    Args:
        location: The city and state, e.g., San Francisco, CA.
    """
    # This is a mock implementation. Replace with a real weather API.
    return f"The weather in {location} is 72°F and sunny."


async def stream_with_tools(question: str, max_iterations: int = 3) -> None:
    """Stream a response from the model, executing any requested tools.

    Args:
        question: The user's question.
        max_iterations: Maximum number of tool-call loops to prevent infinite loops.
    """
    model = init_chat_model(
        "ollama/llama3.1",
        temperature=0,
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    model_with_tools = model.bind_tools([get_current_weather])
    messages = [HumanMessage(content=question)]

    for _ in range(max_iterations):
        full_response = None
        async for chunk in model_with_tools.astream(messages):
            if full_response is None:
                full_response = chunk
            else:
                full_response = full_response + chunk
            if chunk.content:
                print(chunk.content, end="", flush=True)
        print()

        if full_response is None:
            break

        if not full_response.tool_calls:
            break

        messages.append(full_response)
        for tool_call in full_response.tool_calls:
            print(f"Calling tool: {tool_call['name']}({tool_call['args']})")
            tool_result = get_current_weather.invoke(tool_call["args"])
            messages.append(
                ToolMessage(content=tool_result, tool_call_id=tool_call["id"])
            )


async def main() -> None:
    await stream_with_tools("What is the weather in San Francisco, CA?")


if __name__ == "__main__":
    asyncio.run(main())
