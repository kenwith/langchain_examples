"""Example 47: Ollama Tools
========================

| Name | Value |
|------|-------|
| Title | Bind simple tools to an Ollama model and execute a tool-calling loop |
| Provider | Ollama |
| Model | llama3.1 (configurable) |
| Description | Demonstrates how to bind tools to a chat model and handle tool calls in a loop. |

This example uses `init_chat_model` to create a model-agnostic chat model,
binds a couple of simple tools, and runs a conversation loop until the model
has no more tool calls to make.
"""

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


def run_tool_calling_loop(model, tools, query: str) -> str:
    """Run a conversation loop, invoking tools whenever the model requests it."""
    messages = [HumanMessage(content=query)]
    max_iterations = 5

    for _ in range(max_iterations):
        response = model.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return response.content

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            matched_tool = next(t for t in tools if t.name == tool_name)
            result = matched_tool.invoke(tool_args)
            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"],
                )
            )

    return "Max iterations reached"


def main() -> None:
    """Run the Ollama tool-calling example."""
    model = init_chat_model(
        model=os.getenv("OLLAMA_MODEL", "llama3.1"),
        model_provider="ollama",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )

    tools = [add, multiply]
    bound_model = model.bind_tools(tools)

    query = "What is 23 * 17? Also add 5 and 7."
    result = run_tool_calling_loop(bound_model, tools, query)
    print(result)


if __name__ == "__main__":
    main()
