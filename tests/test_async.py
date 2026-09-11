"""Tests for the async example.

These tests verify that the async example completes and returns the expected
structure when using a mock model. The model is mocked to avoid real API calls
and to keep the tests fast and deterministic.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

import async_example

# | Test | Description |
# |------|-------------|
# | test_async_example_returns_expected_structure | Verifies the async example returns the expected content. |
# | test_async_example_calls_model_with_prompt | Verifies the async example invokes the model with the correct prompt. |
# | test_async_example_uses_init_chat_model | Verifies the async example uses the provider-agnostic init_chat_model. |
# | test_async_example_with_fake_chat_model | Verifies the async example works with a fake chat model. |


@pytest.mark.asyncio
async def test_async_example_returns_expected_structure() -> None:
    """Test that the async example completes and returns the expected content."""
    mock_model = AsyncMock()
    mock_model.ainvoke.return_value = AIMessage(content="Hello, world!")

    with patch.object(async_example, "init_chat_model", return_value=mock_model):
        result = await async_example.main()

    assert result == "Hello, world!"


@pytest.mark.asyncio
async def test_async_example_calls_model_with_prompt() -> None:
    """Test that the async example invokes the model with the expected prompt."""
    mock_model = AsyncMock()
    mock_model.ainvoke.return_value = AIMessage(content="Hello, world!")

    with patch.object(async_example, "init_chat_model", return_value=mock_model):
        await async_example.main()

    mock_model.ainvoke.assert_awaited_once_with("Hello")


@pytest.mark.asyncio
async def test_async_example_uses_init_chat_model() -> None:
    """Test that the async example uses the provider-agnostic init_chat_model."""
    mock_model = AsyncMock()
    mock_model.ainvoke.return_value = AIMessage(content="Hello, world!")

    with patch.object(async_example, "init_chat_model", return_value=mock_model) as mock_init:
        await async_example.main()

    mock_init.assert_called_once()


@pytest.mark.asyncio
async def test_async_example_with_fake_chat_model() -> None:
    """Test that the async example works with a fake chat model."""
    model = FakeMessagesListChatModel(responses=[AIMessage(content="Hello, world!")])

    with patch.object(async_example, "init_chat_model", return_value=model):
        result = await async_example.main()

    assert result == "Hello, world!"


if __name__ == "__main__":
    pytest.main([__file__])
