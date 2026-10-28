"""
Smoke test for example 85: Ollama async LangGraph adaptive RAG.
Verifies the async graph can be built without network access.
"""

import sys
import os
from unittest.mock import patch

# Ensure the repository root is on the path so the example module can be imported.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import example_85_ollama_async_langgraph_adaptive_rag as example


class DummyChatModel:
    """Placeholder chat model that avoids any network or model loading."""

    def __init__(self, *args, **kwargs):
        pass

    def bind_tools(self, *args, **kwargs):
        return self

    def invoke(self, *args, **kwargs):
        return {"content": "dummy"}

    async def ainvoke(self, *args, **kwargs):
        return {"content": "dummy"}


def test_build_graph():
    """
    Verify that the adaptive RAG graph can be constructed without network calls.

    The example's `init_chat_model` is patched with a dummy model to prevent any
    real LLM or embedding API interaction.
    """
    with patch.object(example, "init_chat_model", return_value=DummyChatModel()):
        graph = example.build_graph()
        assert graph is not None


if __name__ == "__main__":
    test_build_graph()
    print("Smoke test passed.")
