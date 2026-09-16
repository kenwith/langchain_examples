"""
| # | Example | Description |
|---|---------|-------------|
| 19 | LLM Caching | Cache LLM responses with InMemoryCache or SQLiteCache to reduce latency and cost. |

This example demonstrates how to use LangChain's caching layer with both in-memory and
persistent SQLite caches. A helper selects the cache implementation, and the SQLite cache
can share responses across separate runs of this script.
"""

import os
from typing import Optional

try:
    from langchain.cache import InMemoryCache, SQLiteCache
except ImportError:
    from langchain_community.cache import InMemoryCache, SQLiteCache

from langchain.chat_models import init_chat_model


def select_cache(cache_type: Optional[str] = None):
    """Return a LangChain cache instance based on user input or environment."""
    cache_type = (cache_type or os.getenv("CACHE_TYPE", "memory")).lower()

    if cache_type == "sqlite":
        db_path = os.getenv("CACHE_DB_PATH", ".langchain_cache.db")
        print(f"Using SQLiteCache (persistent) at: {db_path}")
        return SQLiteCache(database_path=db_path)

    print("Using InMemoryCache (non-persistent)")
    return InMemoryCache()


def demonstrate_caching(cache_type: Optional[str] = None):
    """Demonstrate LLM response caching with the selected cache backend."""
    # Use environment variables for model and provider to remain provider-agnostic.
    # Example: set OPENAI_API_KEY, ANTHROPIC_API_KEY, etc. in your environment.
    model = os.getenv("MODEL", "gpt-4o-mini")
    provider = os.getenv("MODEL_PROVIDER", "openai")

    # Initialize the chat model using init_chat_model (provider-agnostic).
    llm = init_chat_model(model, model_provider=provider, temperature=0)

    # Enable caching with the selected backend.
    cache = select_cache(cache_type)
    llm.cache = cache

    prompt = "What is the capital of France?"

    print("First call (cache miss on a fresh cache, or possible hit with a persistent cache):")
    first_response = llm.invoke(prompt)
    print(first_response)

    print("\nSecond call (guaranteed cache hit):")
    second_response = llm.invoke(prompt)
    print(second_response)

    # Verify that the responses are identical, proving the cache was used.
    assert first_response == second_response, "Cached response should match the original."
    print("\n✅ Cache verified: identical response returned without a new API call.")


if __name__ == "__main__":
    # Example: CACHE_TYPE=sqlite python examples/19_llm_caching.py
    demonstrate_caching()
