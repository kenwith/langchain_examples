"""Example 11: Parallel Tool Calls.

This example demonstrates how to use `bind_tools` and a custom
`dispatch_parallel_tool_calls` helper to execute multiple tool calls in
parallel, with consistent logging for tool failures.

| Example | Description |
|---------|-------------|
| 11 | Parallel tool calls with `bind_tools` and a custom dispatch helper. |
"""

import logging
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Annotated, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


class State(TypedDict):
    messages: Annotated[list, add_messages]


def dispatch_parallel_tool_calls(tool_calls, tools, logger=None):
    """Execute multiple tool calls in parallel.

    Each tool call is executed concurrently. Failures are logged and returned
    as error ToolMessages so the agent can recover gracefully.

    Args:
        tool_calls: List of tool call dicts from an AIMessage.
        tools: List of LangChain tools.
        logger: Optional logger; defaults to module logger.

    Returns:
        List[ToolMessage] in the same order as tool_calls.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    tool_by_name = {tool.name: tool for tool in tools}
    results = [None] * len(tool_calls)

    def execute_tool_call(index, tool_call):
        name = tool_call.get("name")
        args = tool_call.get("args", {})
        call_id = tool_call.get("id") or f"call_{index}"
        logger.info("Dispatching tool call %s: %s(%s)", call_id, name, args)

        tool = tool_by_name.get(name)
        if tool is None:
            error_msg = f"Tool '{name}' not found."
            logger.error("Tool call %s failed: %s", call_id, error_msg)
            return ToolMessage(
                content=error_msg,
                tool_call_id=call_id,
                name=name,
                status="error",
            )

        try:
            result = tool.invoke(args)
            logger.info("Tool call %s succeeded: %s", call_id, result)
            return ToolMessage(
                content=str(result),
                tool_call_id=call_id,
                name=name,
            )
        except Exception as exc:
            logger.error(
                "Tool call %s failed with exception: %s",
                call_id,
                exc,
                exc_info=True,
            )
            return ToolMessage(
                content=f"Error: {exc}",
                tool_call_id=call_id,
                name=name,
                status="error",
            )

    with ThreadPoolExecutor(max_workers=max(1, len(tool_calls))) as executor:
        future_to_index = {
            executor.submit(execute_tool_call, i, tc): i
            for i, tc in enumerate(tool_calls)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            results[index] = future.result()

    return results


def create_graph():
    """Create a LangGraph agent that can execute parallel tool calls."""
    # Provider-agnostic model initialization.
    # Set CHAT_MODEL and CHAT_MODEL_PROVIDER to use a different model/provider.
    model = init_chat_model(
        model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
        model_provider=os.getenv("CHAT_MODEL_PROVIDER", "openai"),
        temperature=0,
    )

    # Detect whether the model supports parallel tool calls.
    parallel_supported = getattr(model, "parallel_tool_calls", True)
    if not parallel_supported:
        warnings.warn(
            "The selected model does not support parallel tool calls. "
            "Falling back to sequential execution."
        )

    tools = [add, multiply]
    model_with_tools = model.bind_tools(tools)

    def call_model(state):
        return {"messages": [model_with_tools.invoke(state["messages"])]}

    def call_tools(state):
        last_message = state["messages"][-1]
        tool_messages = dispatch_parallel_tool_calls(
            last_message.tool_calls, tools
        )
        return {"messages": tool_messages}

    def should_continue(state):
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tools"
        return END

    graph = StateGraph(State)
    graph.add_node("agent", call_model)
    graph.add_node("tools", call_tools)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "agent")

    return graph.compile()


def main():
    """Run a simple demo of parallel tool calls."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    app = create_graph()

    # This prompt should trigger two independent tool calls in one response.
    result = app.invoke(
        {"messages": [("human", "What is 2 + 3 and 4 * 5?")]}
    )

    for message in result["messages"]:
        if hasattr(message, "pretty_print"):
            message.pretty_print()
        else:
            print(message)


if __name__ == "__main__":
    main()
