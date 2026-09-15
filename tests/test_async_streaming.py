"""Unit tests for the async streaming helpers using mocked model responses."""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_examples.streaming import stream_async  # Adjust import as needed


# Test: stream_async yields tokens as they arrive
def test_stream_async_yields_tokens():
    """Verify that stream_async yields tokens from the model's async stream."""
    # Create a mock model with an astream method that yields chunks
    mock_model = MagicMock()
    async def mock_astream(messages, **kwargs):
        for token in ["Hello", " ", "world", "!"]:
            yield token

    mock_model.astream = mock_astream

    # Patch init_chat_model to return the mock
    with patch("langchain_examples.streaming.init_chat_model", return_value=mock_model):
        result = asyncio.run(stream_async("Test prompt"))
        assert result == ["Hello", " ", "world", "!"]


# Test: stream_async handles empty stream
def test_stream_async_empty_stream():
    """Verify that stream_async returns an empty list when the model yields nothing."""
    mock_model = MagicMock()
    async def mock_astream(messages, **kwargs):
        return  # yields nothing

    mock_model.astream = mock_astream

    with patch("langchain_examples.streaming.init_chat_model", return_value=mock_model):
        result = asyncio.run(stream_async("Test prompt"))
        assert result == []


# Test: stream_async propagates exceptions from the model
def test_stream_async_handles_exception():
    """Verify that stream_async re-raises exceptions from the model's stream."""
    mock_model = MagicMock()
    async def mock_astream(messages, **kwargs):
        raise RuntimeError("Model failure")

    mock_model.astream = mock_astream

    with patch("langchain_examples.streaming.init_chat_model", return_value=mock_model):
        with pytest.raises(RuntimeError, match="Model failure"):
            asyncio.run(stream_async("Test prompt"))


# Test: stream_async passes correct messages to the model
def test_stream_async_passes_messages():
    """Verify that stream_async constructs and passes the correct message list."""
    mock_model = MagicMock()
    captured_messages = []

    async def mock_astream(messages, **kwargs):
        captured_messages.extend(messages)
        yield "token"

    mock_model.astream = mock_astream

    with patch("langchain_examples.streaming.init_chat_model", return_value=mock_model):
        asyncio.run(stream_async("Hello", system="You are a helpful assistant."))

    # Check that the messages list contains the system and user messages
    assert len(captured_messages) == 2
    assert captured_messages[0].type == "system"
    assert captured_messages[0].content == "You are a helpful assistant."
    assert captured_messages[1].type == "human"
    assert captured_messages[1].content == "Hello"


# Test: stream_async supports additional model parameters
def test_stream_async_with_kwargs():
    """Verify that stream_async passes additional kwargs to the model."""
    mock_model = MagicMock()
    async def mock_astream(messages, **kwargs):
        assert kwargs.get("temperature") == 0.5
        yield "token"

    mock_model.astream = mock_astream

    with patch("langchain_examples.streaming.init_chat_model", return_value=mock_model):
        result = asyncio.run(stream_async("Test", temperature=0.5))
        assert result == ["token"]


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__]))
