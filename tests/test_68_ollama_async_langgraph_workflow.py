"""Tests for the async Ollama LangGraph workflow example."""

from __future__ import annotations

import asyncio
import os
import socket
import sys
from pathlib import Path

import pytest

# Ensure the repo root is on the path so that `examples` is importable.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples import ollama_async_langgraph_workflow as example  # noqa: E402


def _ollama_available() -> bool:
    """Return True if an Ollama server is reachable on localhost:11434."""
    try:
        with socket.create_connection(("localhost", 11434), timeout=2):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not _ollama_available(),
    reason="Ollama server is not available on localhost:11434",
)

# -----------------------------------------------------------------------------
# Test: Async Ollama LangGraph Workflow
# -----------------------------------------------------------------------------
# | Field       | Value                                                  |
# |-------------|--------------------------------------------------------|
# | Test File   | tests/test_68_ollama_async_langgraph_workflow.py        |
# | Description | Verifies the async Ollama LangGraph workflow runs.      |
# | Requires    | Ollama running locally on port 11434                    |
# -----------------------------------------------------------------------------


def test_ollama_async_langgraph_workflow() -> None:
    """Run the async Ollama LangGraph workflow and assert it completes."""
    # Ensure Ollama-related environment variables are set for the example.
    os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
    os.environ.setdefault("OLLAMA_MODEL", os.getenv("OLLAMA_MODEL", "llama3.1"))

    # The example uses provider-agnostic init_chat_model internally and
    # exposes a `run` coroutine that executes the workflow.
    result = asyncio.run(example.run())

    assert result is not None
    assert isinstance(result, str) and len(result) > 0


if __name__ == "__main__":
    # Small demo block for direct execution.
    if not _ollama_available():
        print("Ollama is not available; skipping demo.")
    else:
        test_ollama_async_langgraph_workflow()
        print("Async Ollama LangGraph workflow test passed.")
