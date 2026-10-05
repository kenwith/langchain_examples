"""Example 55: Ollama Fallbacks.

This example demonstrates how to configure fallback models for a local Ollama
request, so that if the primary model is unavailable or fails, the request can
degrade gracefully to a backup model.

+------------------+---------------------------------------------------+
| Key              | Value                                             |
+==================+===================================================+
| Primary model    | llama3.2 via Ollama                               |
| Fallback model   | gpt-4o-mini via OpenAI (if key present)           |
|                  | otherwise llama3.1 via Ollama                     |
| Provider         | Ollama (primary), OpenAI/Ollama (fallback)        |
+------------------+---------------------------------------------------+

Prerequisites:
- Install ``langchain``, ``langchain-ollama``, and optionally ``langchain-openai``.
- Start Ollama locally and pull the primary model (e.g., ``ollama pull llama3.2``).
- Set ``OPENAI_API_KEY`` if you want to use OpenAI as fallback.

Usage:
    python 55_ollama_fallbacks.py
"""

import os

from langchain.chat_models import init_chat_model


def get_primary_model():
    """Return the primary Ollama chat model."""
    return init_chat_model("llama3.2", provider="ollama", temperature=0)


def get_fallback_model():
    """Return a fallback chat model.

    Prefers OpenAI if an API key is available; otherwise falls back to another
    local Ollama model.
    """
    if os.getenv("OPENAI_API_KEY"):
        return init_chat_model("gpt-4o-mini", provider="openai", temperature=0)

    return init_chat_model("llama3.1", provider="ollama", temperature=0)


def create_model_with_fallbacks():
    """Create a chat model with fallback support."""
    primary = get_primary_model()
    fallback = get_fallback_model()
    return primary.with_fallbacks([fallback])


def run_demo():
    """Run a simple query with fallback configured."""
    model = create_model_with_fallbacks()
    response = model.invoke(
        "Explain the concept of graceful degradation in one sentence."
    )
    print(response.content)


if __name__ == "__main__":
    run_demo()
