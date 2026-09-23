"""Tests for LangGraph subgraphs in the langchain_examples repo.

This module demonstrates how to compile parent and child graphs and assert the composed workflow returns the expected state. It follows the repo conventions:
- provider-agnostic `init_chat_model` usage
- plain functions for graph nodes
- README-style section headers
- a small `__main__` demo block

Credentials are referenced only via `os.getenv()` with placeholder names; never hardcode secrets.
"""

import os
from typing import TypedDict

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, END, START

# -----------------------------------------------------------------------------
# State schema
# -----------------------------------------------------------------------------


class GraphState(TypedDict):
    """State passed through the graphs."""
    messages: list
    result: str


# -----------------------------------------------------------------------------
# Graph node functions
# -----------------------------------------------------------------------------


def child_node(state: GraphState) -> GraphState:
    """Child graph node: appends a message to the state."""
    return {**state, "messages": state["messages"] + ["Child processed"]}


def final_node(state: GraphState) -> GraphState:
    """Parent graph final node: sets the result."""
    return {**state, "result": "Done"}


# -----------------------------------------------------------------------------
# Graph builders
# -----------------------------------------------------------------------------


def build_child_graph(model) -> StateGraph:
    """Build and compile the child graph."""
    graph = StateGraph(GraphState)
    graph.add_node("child_node", child_node)
    graph.add_edge(START, "child_node")
    graph.add_edge("child_node", END)
    return graph.compile()


def build_parent_graph(model, child_graph) -> StateGraph:
    """Build and compile the parent graph with the child as a subgraph."""
    graph = StateGraph(GraphState)

    # Node that invokes the child graph (closure captures child_graph)
    def call_child(state: GraphState) -> GraphState:
        return child_graph.invoke(state)

    graph.add_node("call_child", call_child)
    graph.add_node("final", final_node)
    graph.add_edge(START, "call_child")
    graph.add_edge("call_child", "final")
    graph.add_edge("final", END)
    return graph.compile()


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


# Test: compile child graph
def test_compile_child_graph():
    """Verify the child graph compiles without errors."""
    model = init_chat_model(model="gpt-4", api_key=os.getenv("OPENAI_API_KEY"))
    child = build_child_graph(model)
    assert child is not None


# Test: compile parent graph
def test_compile_parent_graph():
    """Verify the parent graph compiles without errors."""
    model = init_chat_model(model="gpt-4", api_key=os.getenv("OPENAI_API_KEY"))
    child = build_child_graph(model)
    parent = build_parent_graph(model, child)
    assert parent is not None


# Test: composed workflow
def test_composed_workflow():
    """Verify the parent+child graph produces the expected state."""
    model = init_chat_model(model="gpt-4", api_key=os.getenv("OPENAI_API_KEY"))
    child = build_child_graph(model)
    parent = build_parent_graph(model, child)

    initial_state: GraphState = {"messages": ["Hello"], "result": ""}
    result = parent.invoke(initial_state)

    assert "messages" in result
    assert "result" in result
    assert len(result["messages"]) == 2, "Expected two messages: initial + child"
    assert result["messages"][-1] == "Child processed"
    assert result["result"] == "Done"


# -----------------------------------------------------------------------------
# Demo block (run when executed directly)
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    # Run all test functions and print a summary
    tests = [
        test_compile_child_graph,
        test_compile_parent_graph,
        test_composed_workflow,
    ]
    for test in tests:
        test()
        print(f"PASSED: {test.__name__}")
    print("All tests passed successfully.")
    print("Note: Ensure OPENAI_API_KEY is set in the environment if using a real model.")
