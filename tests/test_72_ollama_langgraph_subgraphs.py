"""Tests for the Ollama LangGraph subgraphs example."""

# ---------------------------------------------------------------------------
# | Test ID | Example                    | Description                         |
# |---------|----------------------------|-------------------------------------|
# | 72      | ollama_langgraph_subgraphs | Test subgraph execution with Ollama |
# ---------------------------------------------------------------------------

import pytest
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage


def test_ollama_langgraph_subgraphs():
    """Test the Ollama subgraphs example."""
    try:
        model = init_chat_model("ollama", model="llama3", temperature=0)
        model.invoke("ping")
    except Exception:
        pytest.skip("Ollama is not available")

    from examples.ollama_langgraph_subgraphs import build_graph

    graph = build_graph()
    result = graph.invoke({"messages": [HumanMessage(content="Hello")]})

    assert "messages" in result
    assert len(result["messages"]) > 1
    assert isinstance(result["messages"][-1].content, str)


if __name__ == "__main__":
    test_ollama_langgraph_subgraphs()
    print("Ollama subgraphs test passed.")
