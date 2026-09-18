"""Test suite for streaming output.

| Test                    | Description                             |
|-------------------------|-----------------------------------------|
| test_stream_incremental | Verifies stream yields multiple chunks. |
| test_stream_content     | Verifies streamed content is non-empty. |
"""

import os

import pytest
from langchain.chat_models import init_chat_model

MODEL = os.getenv("CHAT_MODEL", "openai:gpt-4o-mini")


def _missing_api_key():
    """Return True if the provider's API key is missing from the environment."""
    if ":" not in MODEL:
        return False
    provider = MODEL.split(":")[0].lower()
    env_var = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "google_genai": "GOOGLE_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "groq": "GROQ_API_KEY",
        "cohere": "COHERE_API_KEY",
    }.get(provider)
    return bool(env_var) and os.getenv(env_var) is None


def _stream_chunks(prompt: str):
    model = init_chat_model(model=MODEL, temperature=0)
    return list(model.stream(prompt))


@pytest.mark.skipif(_missing_api_key(), reason="API key not set")
def test_stream_incremental():
    """Streaming should yield more than one chunk for a sufficiently long prompt."""
    chunks = _stream_chunks("Write a detailed paragraph about the solar system.")
    assert len(chunks) > 1


@pytest.mark.skipif(_missing_api_key(), reason="API key not set")
def test_stream_content():
    """Streaming chunks should combine into a non-empty response."""
    chunks = _stream_chunks("Say hello in one sentence.")
    full = "".join(chunk.content for chunk in chunks)
    assert len(full) > 0
    assert any(chunk.content for chunk in chunks)


if __name__ == "__main__":
    print(f"Streaming demo with model: {MODEL}")
    model = init_chat_model(model=MODEL, temperature=0)
    for chunk in model.stream("Tell me a short joke."):
        print(chunk.content, end="", flush=True)
    print()
