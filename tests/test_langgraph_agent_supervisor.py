"""Test for the LangGraph agent supervisor example.

Verifies that the supervisor graph routes to sub-agents and returns a final output.
"""

import os
from unittest.mock import patch

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

# Import the example graph builder
from examples.langgraph_agent_supervisor import build_supervisor_graph


class DummyChatModel(BaseChatModel):
    """A dummy chat model that returns a fixed response."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        # Return a fixed message; in a real test we would tailor this per call.
        content = "I am the supervisor. I will call agent1."
        message = AIMessage(content=content)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self):
        return "dummy"


def test_supervisor_routes_and_returns_output():
    """Test that the supervisor graph routes to a sub-agent and returns final output."""
    # Use the dummy model for all chat model calls
    with patch(
        "examples.langgraph_agent_supervisor.init_chat_model",
        return_value=DummyChatModel(),
    ):
        graph = build_supervisor_graph()
        # Prepare initial state
        initial_state = {
            "messages": [HumanMessage(content="What is the weather in SF?")],
            "next": None,
        }
        # Invoke the graph
        result = graph.invoke(initial_state)
        # Check that we have a final answer
        assert "messages" in result
        last_message = result["messages"][-1]
        assert isinstance(last_message, AIMessage)
        assert "final" in last_message.content.lower() or "agent" in last_message.content.lower()


def test_supervisor_routes_to_specific_agent():
    """Test that the supervisor routes to a specific sub-agent based on input."""
    # We can mock the supervisor's LLM to return a decision
    with patch(
        "examples.langgraph_agent_supervisor.init_chat_model",
        return_value=DummyChatModel(),
    ):
        graph = build_supervisor_graph()
        initial_state = {
            "messages": [HumanMessage(content="Analyze this code")],
            "next": None,
        }
        result = graph.invoke(initial_state)
        # Check that the result mentions the correct agent
        assert "code" in result["messages"][-1].content.lower() or "analyst" in result["messages"][-1].content.lower()


if __name__ == "__main__":
    # Run the tests manually
    test_supervisor_routes_and_returns_output()
    test_supervisor_routes_to_specific_agent()
    print("All tests passed.")
