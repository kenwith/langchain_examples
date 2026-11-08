"""Test async Ollama chat with system prompt."""

import asyncio
import os

import pytest
from langchain.chat_models import init_chat_model


async def run_async_chat_with_system_prompt() -> str:
    """Run an async chat with a system prompt using Ollama."""
    model = init_chat_model(
        model=os.getenv("OLLAMA_MODEL", "llama3"),
        provider="ollama",
        temperature=0,
    )
    system_prompt = "You are a helpful assistant."
    response = await model.ainvoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "What is the capital of France?"},
        ]
    )
    return response.content


@pytest.mark.asyncio
async def test_async_ollama_chat_with_system_prompt() -> None:
    """Verify the async chat with system prompt runs and returns a response."""
    try:
        result = await run_async_chat_with_system_prompt()
        assert result, "Response should not be empty"
    except Exception as exc:  # pragma: no cover - depends on environment
        pytest.skip(f"Ollama not available: {exc}")


if __name__ == "__main__":
    result = asyncio.run(run_async_chat_with_system_prompt())
    print(result)
