"""Tests for example 102: Ollama chat with a system prompt.

| Field    | Value                                                      |
|----------|------------------------------------------------------------|
| Example  | 102_ollama_chat_with_system_prompt                         |
| Purpose  | Verify the example runs and returns the expected structure |
| Provider | Ollama, invoked through provider-agnostic init_chat_model  |
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"
EXAMPLE_MODULE = "102_ollama_chat_with_system_prompt"


def _load_example() -> Any:
    """Import the example module from the examples directory."""
    if str(EXAMPLES_DIR) not in sys.path:
        sys.path.insert(0, str(EXAMPLES_DIR))
    return importlib.import_module(EXAMPLE_MODULE)


def test_example_imports_cleanly() -> None:
    """The example module imports without raising."""
    module = _load_example()
    assert module is not None


def test_build_messages_returns_system_and_user() -> None:
    """The message builder returns a system message followed by a user message."""
    module = _load_example()
    messages = module.build_messages("be terse", "hello")
    assert isinstance(messages, list)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "be terse"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "hello"


def test_chat_returns_expected_structure(monkeypatch: pytest.MonkeyPatch) -> None:
    """The chat helper returns a dict with the documented keys."""
    module = _load_example()

    class _StubResponse:
        content = "stubbed reply"

    class _StubModel:
        def invoke(self, messages: Any) -> _StubResponse:
            return _StubResponse()

    def _stub_init_chat_model(*args: Any, **kwargs: Any) -> _StubModel:
        return _StubModel()

    monkeypatch.setattr(module, "init_chat_model", _stub_init_chat_model)

    result = module.chat("be terse", "hello")
    assert isinstance(result, dict)
    assert set(result.keys()) >= {"system_prompt", "user_prompt", "response"}
    assert result["system_prompt"] == "be terse"
    assert result["user_prompt"] == "hello"
    assert result["response"] == "stubbed reply"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
