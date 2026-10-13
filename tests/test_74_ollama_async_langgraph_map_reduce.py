"""Tests for the Ollama async LangGraph map-reduce example."""

# | Test | Description |
# |------|-------------|
# | test_map_reduce_example_runs | Validates the Ollama async LangGraph map-reduce example executes |

import asyncio
import importlib
import os
import sys
from pathlib import Path

import pytest

# Ensure the repository root is on sys.path so the example package can be imported.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXAMPLE_MODULE = "examples.ollama_async_langgraph_map_reduce"


def _ollama_available() -> bool:
    """Return True if Ollama is reachable and a model can be invoked.

    Uses provider-agnostic init_chat_model to verify the configured Ollama
    model responds to a simple prompt.
    """
    try:
        from langchain.chat_models import init_chat_model

        model_name = os.getenv("OLLAMA_MODEL", "llama3.1")
        model = init_chat_model(model=model_name, provider="ollama")
        response = model.invoke("ping")
        return response is not None
    except Exception:
        return False


OLLAMA_AVAILABLE = _ollama_available()


def _get_example_entrypoint():
    """Return the async entrypoint callable from the example module."""
    module = importlib.import_module(EXAMPLE_MODULE)
    for name in ("run", "main", "run_example"):
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(
        f"{EXAMPLE_MODULE} must expose run(), main(), or run_example()"
    )


@pytest.mark.skipif(not OLLAMA_AVAILABLE, reason="Ollama server or model is not available")
def test_map_reduce_example_runs():
    """Validate that the Ollama async LangGraph map-reduce example executes."""
    entrypoint = _get_example_entrypoint()
    result = asyncio.run(entrypoint())
    assert result is not None


if __name__ == "__main__":
    if not OLLAMA_AVAILABLE:
        print("Ollama is not available; skipping demo.")
    else:
        entrypoint = _get_example_entrypoint()
        result = asyncio.run(entrypoint())
        print(result)
