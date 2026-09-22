"""Example 11: Parallel Tool Calls.

This example demonstrates how to use `bind_tools` and a custom
`dispatch_parallel_tool_calls` helper to execute multiple tool calls in
parallel, with consistent logging for tool failures.

Execution flow:
1. The user provides a prompt that requires multiple independent tool calls.
2. The agent model is invoked with the prompt; because tools are bound, it
   may return one or more tool calls in a single response.
3. The `call_tools` node dispatches all tool calls concurrently using a
   ThreadPoolExecutor.
4. Each tool call is executed independently; failures are captured as error
   ToolMessages.
5. The tool results are sorted by tool name and combined into a summary
   SystemMessage.
6. The agent receives the tool results and the summary, then generates a
   final answer.

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
from langchain_core.messages import SystemMessage, ToolMessage
# Use the `tool` decorator from `langchain_core.tools` (not the legacy
# `Tool` class from `langchain.tools`) to define tools.
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
    as error ToolMessages so the agent can recover gracefully. Tool calls are
    sorted by name before execution for deterministic result ordering.

    Args:
        tool_calls: List of tool call dicts from an AIMessage.
        tools: List of LangChain tools.
        logger: Optional logger; defaults to module logger.

    Returns:
        List[ToolMessage] sorted by tool name.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    tool_by_name = {tool.name: tool for tool in tools}

    # Sort tool calls by name for deterministic output ordering. The actual
    # execution order is concurrent and may differ, but the returned messages
    # will be in a stable, sorted order.
    sorted_tool_calls = sorted(
        tool_calls, key=lambda tc: tc.get("name") or ""
    )
    results = [None] * len(sorted_tool_calls)

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

    # Run tool calls concurrently. The ThreadPoolExecutor submits all calls
    # and `as_completed` yields futures as they finish. Results are placed
    # into the preallocated list by the original sorted index, so the final
    # order is deterministic even though execution is concurrent.
    with ThreadPoolExecutor(max_workers=max(1, len(sorted_tool_calls))) as executor:
        future_to_index = {
            executor.submit(execute_tool_call, i, tc): i
            for i, tc in enumerate(sorted_tool_calls)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            results[index] = future.result()

    return results


def sort_and_combine_tool_results(tool_messages):
    """Sort tool messages by tool name and combine into a single summary string.

    This helper provides a concise overview of all parallel tool results,
    which is injected into the conversation before the model generates its
    final answer.

    Args:
        tool_messages: List of ToolMessage instances.

    Returns:
        A single string summarizing each tool result in sorted order.
    """
    sorted_messages = sorted(tool_messages, key=lambda msg: msg.name or "")
    lines = [f"- {msg.name}: {msg.content}" for msg in sorted_messages]
    return "Combined tool results:\n" + "\n".join(lines)


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

        # Add a combined, sorted summary of all tool results as a system
        # message to give the model a concise overview before its final answer.
        combined_summary = sort_and_combine_tool_results(tool_messages)
        summary_message = SystemMessage(content=combined_summary)

        return {"messages": tool_messages + [summary_message]}

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
