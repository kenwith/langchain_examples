"""Example: Async Streaming with Tool Calls using Ollama.

This example demonstrates how to use LangChain's `init_chat_model` function to
create a chat model with Ollama as the provider, and then use async streaming to
handle tool calls in real-time. The model is configured to use a tool that
returns the current weather for a given city.

The example shows:
- How to define a tool using the `@tool` decorator.
- How to create a chat model with tool binding.
- How to stream the model's responses asynchronously.
- How to process tool calls and continue the conversation.

To run this example, ensure that Ollama is running locally and that the
`OLLAMA_BASE_URL` environment variable is set if not using the default.
"""

import asyncio
import os

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool


@tool
def get_current_weather(city: str) -> str:
    """Get the current weather for a city."""
    # This is a mock implementation. Replace with a real API call.
    return f"The weather in {city} is sunny with a temperature of 25°C."


async def run_async_streaming_with_tools() -> None:
    """Run an async streaming conversation with tool calls."""
    # Initialize the chat model with Ollama.
    # The model name is "llama3.1" but can be changed.
    # Use the OLLAMA_BASE_URL environment variable if Ollama is not on localhost.
    model = init_chat_model(
        model="llama3.1",
        provider="ollama",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0,
    )

    # Bind the tool to the model.
    model_with_tools = model.bind_tools([get_current_weather])

    # Create a list of messages to start the conversation.
    messages = [HumanMessage(content="What is the weather in Paris?")]
    print("Human:", messages[0].content)

    # Loop to handle multiple tool-call rounds.
    while True:
        # Stream the response asynchronously.
        full_response = None
        async for chunk in model_with_tools.astream(messages):
            # Accumulate the chunks to reconstruct the full AI message.
            if full_response is None:
                full_response = chunk
            else:
                full_response += chunk
            # Print the content as it streams.
            if chunk.content:
                print(chunk.content, end="", flush=True)
        print()  # newline after streaming

        # If there are no tool calls, we are done.
        if not full_response.tool_calls:
            break

        # Append the AI message with tool calls to the conversation.
        messages.append(full_response)

        # Execute each tool call and append the result as a ToolMessage.
        for tool_call in full_response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            print(f"  -> Calling tool: {tool_name}({tool_args})")
            if tool_name == "get_current_weather":
                result = get_current_weather.invoke(tool_args)
            else:
                result = "Unknown tool"
            messages.append(
                ToolMessage(content=str(result), tool_call_id=tool_call["id"])
            )


async def main() -> None:
    """Main entry point for the example."""
    await run_async_streaming_with_tools()


if __name__ == "__main__":
    asyncio.run(main())
