"""
Example 48: Ollama Parallel Tool Calls
======================================

This example demonstrates how to bind multiple tools to an Ollama chat model and
execute the resulting tool calls in parallel. It uses the provider-agnostic
`init_chat_model` function to create the model, and plain Python functions as tools.

Key concepts:
- Binding tools to a chat model with `bind_tools`
- Handling multiple tool calls in a single response
- Executing tool calls concurrently with `ThreadPoolExecutor`

Requirements:
- Install `langchain`, `langchain-core`, `langchain-ollama` (or compatible provider)
- Have an Ollama server running locally or accessible via `OLLAMA_BASE_URL`
"""

from concurrent.futures import ThreadPoolExecutor
import os

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool


@tool
def add_numbers(a: int, b: int) -> int:
    """Add two integers and return the result."""
    return a + b


@tool
def multiply_numbers(a: int, b: int) -> int:
    """Multiply two integers and return the result."""
    return a * b


def main() -> None:
    """Run the parallel tool calling demo."""
    # Initialize the chat model (provider-agnostic, defaults to Ollama)
    model = init_chat_model(
        model=os.getenv("OLLAMA_MODEL", "llama3.1"),
        model_provider="ollama",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )

    # Bind the tools to the model
    model_with_tools = model.bind_tools([add_numbers, multiply_numbers])

    # Prompt that should trigger two tool calls in parallel
    prompt = "What is 3 + 4 and 5 * 6? Please compute both."

    # Invoke the model
    response = model_with_tools.invoke(prompt)

    # Check if there are tool calls in the response
    if not response.tool_calls:
        print("No tool calls were made. Model response:")
        print(response.content)
        return

    print(f"Model generated {len(response.tool_calls)} tool calls.")
    for call in response.tool_calls:
        print(f"  - Tool: {call['name']}, Args: {call['args']}")

    # Execute tool calls in parallel using a thread pool
    with ThreadPoolExecutor() as executor:
        futures = []
        for call in response.tool_calls:
            tool_name = call["name"]
            tool_args = call["args"]
            # Find the corresponding tool function
            tool_func = next(t for t in [add_numbers, multiply_numbers] if t.name == tool_name)
            # Submit the tool call to the executor
            futures.append(executor.submit(tool_func.invoke, tool_args))

        # Collect results
        results = [future.result() for future in futures]

    # Print the results
    print("\nTool execution results:")
    for i, (call, result) in enumerate(zip(response.tool_calls, results)):
        print(f"  {i+1}. {call['name']}({call['args']}) = {result}")


if __name__ == "__main__":
    main()
