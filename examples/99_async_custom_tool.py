"""Example: Async Custom Tool with a Chat Model.

This example demonstrates how to create an asynchronous custom tool and use it
with a chat model via the provider-agnostic init_chat_model function.
"""

import asyncio
import os

from langchain.chat_models import init_chat_model
from langchain.tools import tool


@tool
async def get_current_time(timezone: str) -> str:
    """Returns the current time in the given timezone.

    Args:
        timezone: A valid IANA timezone string (e.g., 'America/New_York').
    """
    # Simulate an async operation, e.g., calling an external API.
    await asyncio.sleep(1)
    # For demonstration, return a static string. In practice, this would call a real service.
    return f"Current time in {timezone}: 12:00 PM"


async def run_demo() -> None:
    """Run the async custom tool example."""
    # Initialize the chat model using the provider-agnostic init_chat_model.
    # The model provider and name can be set via environment variables.
    model = init_chat_model(
        model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
        model_provider=os.getenv("CHAT_MODEL_PROVIDER", "openai"),
    )
    # Bind the tool to the model.
    model_with_tool = model.bind_tools([get_current_time])
    # Create a prompt that asks for the current time.
    messages = [
        ("user", "What is the current time in New York?")
    ]
    # Invoke the model.
    response = await model_with_tool.ainvoke(messages)
    # Print the response.
    print("Model response:", response.content)
    # If the model wants to call the tool, execute it.
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        print(f"Tool called: {tool_name} with args {tool_args}")
        # Execute the tool.
        result = await get_current_time.ainvoke(tool_args)
        print("Tool result:", result)


if __name__ == "__main__":
    asyncio.run(run_demo())
