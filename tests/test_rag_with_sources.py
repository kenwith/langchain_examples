"""Test for the RAG with sources example.

This test verifies that the example script runs successfully and returns
both an answer and a list of source documents.
"""

import os
import sys
from pathlib import Path

import pytest
from langchain.chat_models import init_chat_model

# Ensure the project root is on sys.path so we can import the example.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Skip the entire module if the example is not present.
pytest.importorskip("rag_with_sources")
from rag_with_sources import run_rag_with_sources

# | Test | Description |
# |------|-------------|
# | test_rag_with_sources_returns_answer_and_sources | Runs the RAG example and checks the response contains answer and sources. |


def test_rag_with_sources_returns_answer_and_sources():
    """Run the RAG with sources example and verify its output."""
    # Skip if no API key is available (the example needs a model provider).
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("No API key set; skipping test that requires a live model.")

    # Provider-agnostic model initialization.
    model = init_chat_model("gpt-4o-mini", temperature=0)

    # Execute the example.
    result = run_rag_with_sources("What is LangChain?", model=model)

    # Verify the result contains both an answer and source documents.
    assert "answer" in result, "The result should contain an 'answer' key."
    assert "sources" in result, "The result should contain a 'sources' key."
    assert result["answer"], "The answer should not be empty."
    assert result["sources"], "The sources list should not be empty."


if __name__ == "__main__":
    test_rag_with_sources_returns_answer_and_sources()
    print("RAG with sources test passed.")
