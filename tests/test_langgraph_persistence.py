"""Unit tests for graph construction and state persistence with a mocked chat model.

This module contains tests for building a LangGraph state graph and verifying that
state is correctly persisted across runs. The chat model is mocked to avoid real API calls.

Table of Contents:
- test_graph_construction: Verifies the graph can be built without errors.
- test_state_persistence: Verifies that state is saved and restored correctly.
- demo: Demonstrates running the tests manually.
"""

from unittest.mock import MagicMock, patch

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated


class GraphState(TypedDict):
    """State schema for the graph."""
    messages: Annotated[list, add_messages]


def _build_graph(model):
    """Build a simple graph with a single node that calls the model."""
    def call_model(state):
        response = model.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(GraphState)
    graph.add_node("model", call_model)
    graph.set_entry_point("model")
    graph.add_edge("model", END)
    return graph


def test_graph_construction():
    """Test that the graph can be built with a mocked model."""
    with patch("langchain.chat_models.init_chat_model") as mock_init:
        mock_model = MagicMock()
        mock_init.return_value = mock_model
        # Use the actual init_chat_model to create a model (mocked)
        from langchain.chat_models import init_chat_model
        model = init_chat_model("mock-provider", model="mock-model")
        graph = _build_graph(model)
        assert graph is not None
        # Check that the graph has the expected nodes
        assert "model" in graph.nodes
        assert graph.entry_point == "model"
        assert graph.finish_point == END


def test_state_persistence():
    """Test that state is persisted and restored using MemorySaver."""
    from langchain.chat_models import init_chat_model
    with patch("langchain.chat_models.init_chat_model") as mock_init:
        mock_model = MagicMock()
        # Configure the mock to return a fixed response
        mock_model.invoke.return_value = "Mocked response"
        mock_init.return_value = mock_model
        model = init_chat_model("mock-provider", model="mock-model")

        # Build graph with memory saver
        memory = MemorySaver()
        graph = _build_graph(model)
        app = graph.compile(checkpointer=memory)

        # Run the graph with an initial state
        config = {"configurable": {"thread_id": "1"}}
        initial_state = {"messages": ["Hello"]}
        result = app.invoke(initial_state, config)

        # Verify that the result contains the mocked response
        assert result["messages"][-1] == "Mocked response"

        # Verify that the state is persisted
        persisted_state = app.get_state(config)
        assert persisted_state.values["messages"] == ["Hello", "Mocked response"]

        # Run again with the same thread_id to simulate a conversation
        second_result = app.invoke({"messages": ["How are you?"]}, config)
        assert second_result["messages"][-1] == "Mocked response"
        # The state should now contain all messages
        final_state = app.get_state(config)
        assert final_state.values["messages"] == [
            "Hello",
            "Mocked response",
            "How are you?",
            "Mocked response",
        ]


def demo():
    """Run the tests manually."""
    test_graph_construction()
    test_state_persistence()
    print("All tests passed!")


if __name__ == "__main__":
    demo()
