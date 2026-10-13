"""Tests for the Ollama async fallback chain.

| Test Name | Description |
|-----------|-------------|
| test_ollama_async_fallbacks | Verifies the fallback chain returns a response when the primary model is unavailable. |
"""

import asyncio

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage


def test_ollama_async_fallbacks(monkeypatch):
    """Verify the fallback chain returns a response when the primary is unavailable."""
    primary = init_chat_model(model="llama3.2", provider="ollama")
    fallback = init_chat_model(model="mistral", provider="ollama")

    async def failing_ainvoke(messages, **kwargs):
        raise ConnectionError("Ollama is not available")

    async def fallback_ainvoke(messages, **kwargs):
        return AIMessage(content="Fallback response")

    monkeypatch.setattr(primary, "ainvoke", failing_ainvoke)
    monkeypatch.setattr(fallback, "ainvoke", fallback_ainvoke)

    chain = primary.with_fallbacks([fallback])

    result = asyncio.run(chain.ainvoke([HumanMessage(content="Hello")]))

    assert result.content == "Fallback response"


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
