"""
| # | Example | Description |
|---|---------|-------------|
| 19 | LLM Caching | Cache LLM responses with InMemoryCache to reduce latency and cost. |

This example demonstrates how to use LangChain's InMemoryCache to cache LLM responses,
reducing latency and cost when the same prompt is sent multiple times.
"""

import os

from langchain.cache import InMemoryCache
from langchain.chat_models import init_chat_model


def demonstrate_caching():
    """Demonstrate LLM response caching with InMemoryCache."""
    # Use environment variables for model and provider to remain provider-agnostic.
    # Example: set OPENAI_API_KEY, ANTHROPIC_API_KEY, etc. in your environment.
    model = os.getenv("MODEL", "gpt-4o-mini")
    provider = os.getenv("MODEL_PROVIDER", "openai")

    # Initialize the chat model using init_chat_model (provider-agnostic).
    llm = init_chat_model(model, model_provider=provider, temperature=0)

    # Enable in-memory caching.
    llm.cache = InMemoryCache()

    prompt = "What is the capital of France?"

    print("First call (not cached):")
    first_response = llm.invoke(prompt)
    print(first_response)

    print("\nSecond call (cached):")
    second_response = llm.invoke(prompt)
    print(second_response)

    # Verify that the responses are identical, proving the cache was used.
    assert first_response == second_response, "Cached response should match the original."
    print("\n✅ Cache verified: identical response returned without a new API call.")


if __name__ == "__main__":
    demonstrate_caching()
