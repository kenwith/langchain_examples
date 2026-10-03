"""Tests for the Ollama chat example using a local model.

This module verifies that the Ollama chat example works with a local
model via LangChain's provider-agnostic `init_chat_model` function.

| Test | Description |
|------|-------------|
| test_ollama_chat_response | Verifies a basic chat response is non-empty. |
| test_ollama_chat_with_system | Verifies chat with a system message. |
"""

import os
import pytest
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

DEFAULT_MODEL = "llama3.1"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def _get_chat_model():
    """Create a chat model instance for Ollama."""
    model_name = os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)
    return init_chat_model(
        model=model_name,
        model_provider="ollama",
        base_url=OLLAMA_BASE_URL,
    )


def test_ollama_chat_response():
    """Test that a basic chat response is generated."""
    model = _get_chat_model()
    try:
        response = model.invoke([HumanMessage(content="Say 'hello' in one word")])
    except Exception as exc:
        pytest.skip(f"Ollama not available or model not found: {exc}")
    assert response.content, "Response should not be empty"
    assert isinstance(response.content, str)


def test_ollama_chat_with_system():
    """Test chat with a system message."""
    model = _get_chat_model()
    try:
        response = model.invoke(
            [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content="What is 2+2?"),
            ]
        )
    except Exception as exc:
        pytest.skip(f"Ollama not available or model not found: {exc}")
    assert response.content, "Response should not be empty"


if __name__ == "__main__":
    print("Ollama Chat Demo")
    print("================")
    model = _get_chat_model()
    response = model.invoke([HumanMessage(content="Hello, how are you?")])
    print(f"Model: {model.model_name}")
    print(f"Response: {response.content}")
