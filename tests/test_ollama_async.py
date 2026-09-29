"""Tests for examples/44_ollama_async.py.

These tests mock the async Ollama model to verify the example completes
successfully without requiring a real Ollama server.
"""

import asyncio
import importlib.util
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Load the example module from the examples directory.
EXAMPLE_PATH = Path(__file__).resolve().parent.parent / "examples" / "44_ollama_async.py"
assert EXAMPLE_PATH.exists(), f"Example file not found: {EXAMPLE_PATH}"
spec = importlib.util.spec_from_file_location("ollama_async_example", EXAMPLE_PATH)
ollama_async_example = importlib.util.module_from_spec(spec)
sys.modules["ollama_async_example"] = ollama_async_example
spec.loader.exec_module(ollama_async_example)


# Test: Example runs successfully with a mocked async model
def test_example_runs_with_mocked_ollama():
    """Verify the example completes successfully with a mocked async model."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Mocked response from Ollama"
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    with patch.object(ollama_async_example, "init_chat_model", return_value=mock_llm) as mock_init:
        asyncio.run(ollama_async_example.main())

    mock_init.assert_called_once()
    mock_llm.ainvoke.assert_awaited_once()


# Test: init_chat_model is called with the Ollama provider
def test_init_chat_model_uses_ollama_provider():
    """Verify the example calls init_chat_model with the Ollama provider."""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="Mocked response"))

    with patch.object(ollama_async_example, "init_chat_model", return_value=mock_llm) as mock_init:
        asyncio.run(ollama_async_example.main())

    args, kwargs = mock_init.call_args
    provider_used = args[0] if args else kwargs.get("provider")
    assert "ollama" in str(provider_used).lower()


if __name__ == "__main__":
    test_example_runs_with_mocked_ollama()
    test_init_chat_model_uses_ollama_provider()
    print("All tests passed!")
