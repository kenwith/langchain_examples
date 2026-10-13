"""
Test for example 65: Ollama async agent with memory.

| Test | Description |
|------|-------------|
| test_run_agent_returns_non_empty_response | Verifies that `run_agent` returns a non-empty response. |

This test requires a running Ollama instance and a model configured through environment variables.
"""

import asyncio
import os

import pytest
from langchain.chat_models import init_chat_model

from examples.example_65_ollama_async_agent_with_memory import run_agent


def test_run_agent_returns_non_empty_response():
    """Verify that run_agent returns a non-empty string response."""
    model_name = os.getenv("OLLAMA_MODEL")
    if not model_name:
        pytest.skip("OLLAMA_MODEL environment variable not set")

    provider = os.getenv("CHAT_MODEL_PROVIDER", "ollama")
    model = init_chat_model(model_name, model_provider=provider)

    question = "What is the capital of France?"
    response = asyncio.run(run_agent(question, model))

    assert response, "run_agent should return a non-empty response"
    assert isinstance(response, str)


if __name__ == "__main__":
    model_name = os.getenv("OLLAMA_MODEL")
    if not model_name:
        raise SystemExit("OLLAMA_MODEL environment variable not set")

    provider = os.getenv("CHAT_MODEL_PROVIDER", "ollama")
    model = init_chat_model(model_name, model_provider=provider)

    question = "What is the capital of France?"
    response = asyncio.run(run_agent(question, model))
    print(response)
