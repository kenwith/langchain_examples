"""
Tests for the reflection graph example.

This test verifies that the reflection graph produces a revised answer and
includes a critique in the final state. It uses a mocked chat model to avoid
external API calls and demonstrates provider-agnostic model initialization
via `init_chat_model`.
"""

import pytest
from unittest.mock import patch
from typing import TypedDict, Annotated

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, END

# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

class ReflectionState(TypedDict):
    """State for the reflection graph."""
    input: str
    answer: str
    critique: str
    revised_answer: str


# ---------------------------------------------------------------------------
# Mock model
# ---------------------------------------------------------------------------

class MockReflectionModel(BaseChatModel):
    """A fake chat model that returns canned responses based on system prompts."""

    responses: dict[str, str]

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        # Extract the system prompt from the first message if it exists
        system = ""
        for msg in messages:
            if isinstance(msg, SystemMessage):
                system = msg.content
                break
        # Find a matching response based on keywords in the system prompt
        if "Generate" in system:
            content = self.responses.get("generate", "Initial answer")
        elif "Critique" in system:
            content = self.responses.get("critique", "The answer lacks detail.")
        elif "Revise" in system:
            content = self.responses.get("revise", "Revised answer with more detail.")
        else:
            content = "Unexpected prompt"
        return [{"message": content, "text": content, "type": "ai"}]

    @property
    def _llm_type(self) -> str:
        return "mock-reflection"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_reflection_graph(model):
    """Build the reflection graph using the provided model."""

    def generate_node(state: ReflectionState) -> dict:
        messages = [
            SystemMessage(content="Generate an answer to the input."),
            HumanMessage(content=state["input"]),
        ]
        response = model.invoke(messages)
        return {"answer": response.content}

    def critique_node(state: ReflectionState) -> dict:
        messages = [
            SystemMessage(content="Critique the answer and provide feedback."),
            HumanMessage(content=f"Input: {state['input']}\nAnswer: {state['answer']}"),
        ]
        response = model.invoke(messages)
        return {"critique": response.content}

    def revise_node(state: ReflectionState) -> dict:
        messages = [
            SystemMessage(content="Revise the answer based on the critique."),
            HumanMessage(
                content=(
                    f"Input: {state['input']}\n"
                    f"Original answer: {state['answer']}\n"
                    f"Critique: {state['critique']}"
                )
            ),
        ]
        response = model.invoke(messages)
        return {"revised_answer": response.content}

    graph = StateGraph(ReflectionState)
    graph.add_node("generate", generate_node)
    graph.add_node("critique", critique_node)
    graph.add_node("revise", revise_node)
    graph.set_entry_point("generate")
    graph.add_edge("generate", "critique")
    graph.add_edge("critique", "revise")
    graph.add_edge("revise", END)
    return graph.compile()


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test_reflection_graph_revises_answer():
    """Verify the graph returns a revised answer and includes a critique."""
    mock_responses = {
        "generate": "Initial answer to the problem.",
        "critique": "This answer could be more detailed.",
        "revise": "Revised answer with expanded reasoning.",
    }
    mock_model = MockReflectionModel(responses=mock_responses)

    # Patch init_chat_model to return our mock (provider-agnostic usage)
    with patch("langchain.chat_models.init_chat_model", return_value=mock_model):
        from langchain.chat_models import init_chat_model
        model = init_chat_model("fake-provider/model")  # any string works, patched

    graph = build_reflection_graph(model)

    # Run the graph
    initial_state = {"input": "What is the capital of France?"}
    result = graph.invoke(initial_state)

    # Assertions
    assert "critique" in result, "Critique missing from state"
    assert result["critique"] == "This answer could be more detailed."
    assert "revised_answer" in result, "Revised answer missing from state"
    assert result["revised_answer"] == "Revised answer with expanded reasoning."
    assert result["revised_answer"] != result["answer"], "Revised answer should differ"


# ---------------------------------------------------------------------------
# Demo (executed when running the file directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Run the test function to demonstrate the behavior
    test_reflection_graph_revises_answer()
    print("All assertions passed.")
