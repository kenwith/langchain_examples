"""
Test for example 64: Ollama Async Streaming.

| # | Example | Description |
|---|---------|-------------|
| 64 | Ollama Async Streaming | Verify the async streaming helper yields tokens using a mocked chat model. |
"""

import asyncio
import importlib.util
import inspect
from pathlib import Path
from unittest.mock import Mock, patch


def _load_example():
    """Load the example module from the examples directory."""
    example_path = Path(__file__).resolve().parents[1] / "examples" / "64_ollama_async_streaming.py"
    spec = importlib.util.spec_from_file_location("example_64_ollama_async_streaming", example_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _get_stream_helper(example):
    """Return the first async generator function found in the example module."""
    for name, obj in vars(example).items():
        if inspect.isasyncgenfunction(obj):
            return obj
    raise AttributeError("No async streaming helper found in example module")


async def _collect_tokens(async_gen):
    """Collect tokens from an async generator into a list."""
    return [token async for token in async_gen]


def test_async_streaming_helper_yields_tokens():
    """Verify that the async streaming helper yields tokens from a mocked chat model."""
    example = _load_example()
    stream_tokens = _get_stream_helper(example)

    mock_model = Mock()

    async def fake_astream(prompt):
        for token in ["Hello", " ", "world"]:
            chunk = Mock()
            chunk.content = token
            yield chunk

    mock_model.astream = fake_astream

    with patch.object(example, "init_chat_model", return_value=mock_model):
        tokens = asyncio.run(_collect_tokens(stream_tokens("test prompt")))

    assert tokens == ["Hello", " ", "world"]


if __name__ == "__main__":
    test_async_streaming_helper_yields_tokens()
    print("Test passed.")
