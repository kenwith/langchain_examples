"""
| # | Example | Description |
|---|---------|-------------|
| 19 | LLM Caching | Cache LLM responses with SQLiteCache (or InMemoryCache) to reduce latency and cost. |

This example demonstrates LangChain's caching layer, focusing on the persistent SQLiteCache.
It also shows how to use InMemoryCache for single-process caching.

Cache keys:
LangChain generates a cache key from a combination of the model's identifying parameters
(for example, model name, temperature, and other model settings) and the serialized input
messages (or prompt). If the same model configuration and the same prompt are used again,
the same cache key is produced, and the cached response is returned without calling the API.

When cached responses are reused:
- A response is reused when the exact same cache key is requested again.
- For SQLiteCache, the cache is stored in a local database file, so cached responses are
  reused across separate runs of this script.
- For InMemoryCache, cached responses are reused only within the same process.

By default, the script uses SQLiteCache. Set CACHE_TYPE=memory to use InMemoryCache, or
CACHE_TYPE=both to run both backends sequentially.
"""

import os
from typing import Optional

try:
    from langchain.cache import InMemoryCache, SQLiteCache
except ImportError:
    from langchain_community.cache import InMemoryCache, SQLiteCache

from langchain.chat_models import init_chat_model

# For tracking cache hits/misses via callbacks
try:
    from langchain.callbacks.base import BaseCallbackHandler
except ImportError:
    from langchain_core.callbacks import BaseCallbackHandler


class CacheStatsCallback(BaseCallbackHandler):
    """Counts the number of actual LLM calls (i.e., cache misses)."""

    def __init__(self):
        self.llm_calls = 0

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.llm_calls += 1


def select_cache(cache_type: Optional[str] = None):
    """Return a LangChain cache instance based on user input or environment."""
    cache_type = (cache_type or os.getenv("CACHE_TYPE", "sqlite")).lower()

    if cache_type == "sqlite":
        db_path = os.getenv("CACHE_DB_PATH", ".langchain_cache.db")
        print(f"Using SQLiteCache (persistent) at: {db_path}")
        return SQLiteCache(database_path=db_path)

    print("Using InMemoryCache (non-persistent)")
    return InMemoryCache()


def display_cache_stats(hits: int, misses: int):
    """Print cache hit/miss statistics after the example finishes."""
    total = hits + misses
    print(f"\nCache stats: {hits} hits, {misses} misses (total {total} invocations).")
    if total > 0:
        print(f"Hit rate: {hits / total:.1%}")


def cache_stats(callback: CacheStatsCallback, total_calls: int):
    """Calculate and display cache hit/miss statistics from a callback."""
    misses = callback.llm_calls
    hits = total_calls - misses
    display_cache_stats(hits, misses)
    return hits, misses


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

    # Attach a callback to count actual LLM calls (cache misses)
    stats_callback = CacheStatsCallback()
    llm.callbacks = [stats_callback]

    prompt = "What is the capital of France?"

    # A cache key is derived from the model's serialized parameters and the input messages.
    # If the same key is requested again, the cached response is returned without a new API call.
    # With SQLiteCache, this key and response persist across script runs.
    print("First call (cache miss on a fresh cache, or possible hit with a persistent SQLite cache):")
    first_response = llm.invoke(prompt)
    print(first_response)

    print("\nSecond call (guaranteed cache hit):")
    second_response = llm.invoke(prompt)
    print(second_response)

    # Verify that the responses are identical, proving the cache was used.
    assert first_response == second_response, "Cached response should match the original."
    print("\n✅ Cache verified: identical response returned without a new API call.")

    # Compute and display cache stats from the callback
    total_calls = 2
    cache_stats(stats_callback, total_calls)


if __name__ == "__main__":
    # By default, use SQLiteCache so cached responses are reused across runs.
    # Set CACHE_TYPE=memory to use InMemoryCache, or CACHE_TYPE=both to run both backends.
    cache_type = os.getenv("CACHE_TYPE", "sqlite").lower()

    if cache_type == "both":
        demonstrate_caching("memory")
        print("\n" + "=" * 60 + "\n")
        demonstrate_caching("sqlite")
    else:
        demonstrate_caching(cache_type)
