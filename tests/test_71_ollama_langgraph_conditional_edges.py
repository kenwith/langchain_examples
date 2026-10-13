"""Tests for the Ollama LangGraph conditional edges example.

This test verifies that the example builds and runs a LangGraph with
conditional edges using an Ollama chat model. The test is skipped when
Ollama is not available.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

# Ensure the examples directory is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))

from langchain.chat_models import init_chat_model  # noqa: E402

# ---------------------------------------------------------------------------
# Test: 71 - Ollama LangGraph Conditional Edges
# ---------------------------------------------------------------------------


def _ollama_available() -> bool:
    """Return True if Ollama is available, False otherwise."""
    try:
        model = init_chat_model("ollama", model="llama3.2")
        # A minimal invocation verifies connectivity and model availability.
        model.invoke("ping")
        return True
    except Exception:
        return False


def _load_example_module():
    """Load the example module from the examples directory."""
    example_path = (
        Path(__file__).parent.parent / "examples" / "71_ollama_langgraph_conditional_edges.py"
    )
    spec = importlib.util.spec_from_file_location(
        "ollama_langgraph_conditional_edges", example_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(
    not _ollama_available(),
    reason="Ollama is not available",
)
def test_ollama_langgraph_conditional_edges():
    """Run the Ollama LangGraph conditional edges example end-to-end."""
    module = _load_example_module()
    run_func = getattr(module, "run", None)
    if run_func is None:
        run_func = getattr(module, "main", None)
    assert run_func is not None, "Example module must define a run() or main() function."
    result = run_func()
    assert result is not None


if __name__ == "__main__":
    test_ollama_langgraph_conditional_edges()
    print("Ollama LangGraph conditional edges test passed.")
