"""Unit tests for the memory example.

This module contains tests for the `format_history()` function and the
`main()` flow in `memory.py`. The chat model is mocked to avoid real API
calls.

| Test | Description |
|------|-------------|
| test_format_history_empty | Verifies that an empty history returns an empty string. |
| test_format_history_messages | Verifies that messages are formatted with role and content. |
| test_main_mocked_model | Verifies that `main()` uses the chat model and prints the response. |
"""

import contextlib
import io
from types import SimpleNamespace
from unittest.mock import Mock, patch

from memory import format_history, main


def test_format_history_empty():
    """An empty history should produce an empty string."""
    assert format_history([]) == ""


def test_format_history_messages():
    """Messages should be formatted with role and content."""
    messages = [
        SimpleNamespace(type="human", content="Hello"),
        SimpleNamespace(type="ai", content="Hi there!"),
    ]
    result = format_history(messages)
    assert "Hello" in result
    assert "Hi there!" in result
    assert "human" in result.lower()
    assert "ai" in result.lower()
    assert "\n" in result


def test_main_mocked_model():
    """The main flow should use the mocked chat model and print the response."""
    mock_model = Mock()
    mock_model.invoke.return_value = SimpleNamespace(content="Mock response")

    with patch("memory.init_chat_model", return_value=mock_model) as mock_init:
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            main()

    mock_init.assert_called_once()
    mock_model.invoke.assert_called_once()
    assert "Mock response" in stdout.getvalue()


if __name__ == "__main__":
    test_format_history_empty()
    test_format_history_messages()
    test_main_mocked_model()
    print("All tests passed!")
