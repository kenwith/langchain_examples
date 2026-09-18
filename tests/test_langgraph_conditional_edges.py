"""Unit tests for the conditional routing logic in the LangGraph example.

| Test                            | What it validates                                   |
|---------------------------------|-----------------------------------------------------|
| test_router_directs_to_tools    | Router returns `"tools"` when tools are required.   |
| test_router_directs_to_respond  | Router returns `"respond"` when tools are not needed.|
| test_conditional_graph_compiles | The graph with conditional edges compiles correctly.|
| test_model_initialization_from_env | `init_chat_model` works with provider-agnostic env vars. |
"""

import os
from typing import Literal, TypedDict

import pytest
from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph

try:
    from langgraph_conditional_edges import should_use_tools
except ImportError:
    from langchain_examples.langgraph_conditional_edges import should_use_tools


class _State(TypedDict):
    question: str
    needs_tools: bool


def _build_test_graph():
    """Return a compiled StateGraph using the example router."""
    graph = StateGraph(_State)
    graph.add_node("router", lambda state: state)
    graph.add_node("respond", lambda state: state)
    graph.add_node("tools", lambda state: state)
    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        should_use_tools,
        {"respond": "respond", "tools": "tools"},
    )
    graph.add_edge("respond", END)
    graph.add_edge("tools", END)
    return graph.compile()


def test_router_directs_to_tools():
    state = _State(question="What's the weather?", needs_tools=True)
    assert should_use_tools(state) == "tools"


def test_router_directs_to_respond():
    state = _State(question="Hello", needs_tools=False)
    assert should_use_tools(state) == "respond"


def test_conditional_graph_compiles():
    compiled = _build_test_graph()
    assert compiled is not None


def test_model_initialization_from_env():
    model_name = os.getenv("LANGGRAPH_CHAT_MODEL", "gpt-4o-mini")
    model_provider = os.getenv("LANGGRAPH_CHAT_PROVIDER")
    try:
        if model_provider:
            model = init_chat_model(model_name, model_provider=model_provider)
        else:
            model = init_chat_model(model_name)
    except Exception as exc:
        pytest.skip(f"init_chat_model not available: {exc}")
    else:
        assert model is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
