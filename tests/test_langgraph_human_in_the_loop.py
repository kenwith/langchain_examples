"""
Tests for the human-in-the-loop LangGraph example.

These tests verify that the graph compiles, can be resumed with a command,
and that the approved path completes successfully.
"""

import os
import pytest

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, END

# -----------------------------------------------------------------------------
# Test: test_compile_graph
# -----------------------------------------------------------------------------
def test_compile_graph():
    """Verify that the human-in-the-loop graph compiles without errors."""
    from langgraph_examples.human_in_the_loop import build_graph

    graph = build_graph()
    assert graph is not None
    # Compile the graph to ensure it's valid
    compiled = graph.compile()
    assert compiled is not None

# -----------------------------------------------------------------------------
# Test: test_resume_approved_path
# -----------------------------------------------------------------------------
def test_resume_approved_path():
    """Invoke the graph with a resume command and assert approved path completes."""
    from langgraph_examples.human_in_the_loop import build_graph

    graph = build_graph()
    compiled = graph.compile()

    # Initial state with a user question and no approval yet
    initial_state = {
        "question": "What is LangGraph?",
        "approval": None,
        "answer": None,
    }

    # Run the graph until it pauses for human approval
    result = compiled.invoke(initial_state)
    # The graph should pause and not have an answer yet
    assert result["answer"] is None
    # There should be a pending approval request
    assert "approval_request" in result

    # Now resume with an approval command (e.g., "yes")
    resume_state = result.copy()
    resume_state["approval"] = "yes"  # or use a command structure
    final_result = compiled.invoke(resume_state)

    # The approved path should produce an answer
    assert final_result["answer"] is not None
    assert "LangGraph" in final_result["answer"]

# -----------------------------------------------------------------------------
# Test: test_resume_denied_path
# -----------------------------------------------------------------------------
def test_resume_denied_path():
    """Invoke the graph with a denial command and ensure it does not produce an answer."""
    from langgraph_examples.human_in_the_loop import build_graph

    graph = build_graph()
    compiled = graph.compile()

    initial_state = {
        "question": "What is LangGraph?",
        "approval": None,
        "answer": None,
    }

    result = compiled.invoke(initial_state)
    assert result["answer"] is None

    resume_state = result.copy()
    resume_state["approval"] = "no"
    final_result = compiled.invoke(resume_state)

    # Denied path should not produce an answer
    assert final_result["answer"] is None

# -----------------------------------------------------------------------------
# Main demo block
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Use provider-agnostic init_chat_model (ensure API key via environment)
    model = init_chat_model(
        "gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0,
    )
    print("Human-in-the-loop demo (tests only)")
    print("Run pytest to execute the tests.")
    # Optionally run a quick smoke test
    from langgraph_examples.human_in_the_loop import build_graph
    graph = build_graph()
    compiled = graph.compile()
    print("Graph compiled successfully.")
