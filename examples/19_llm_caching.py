"""
Example: LLM Caching with Hit/Miss Counting

This example demonstrates how to use LangChain's LLM caching mechanism and
track cache hits and misses with a custom counting cache.
"""

import os

from langchain.cache import BaseCache, InMemoryCache, set_llm_cache
from langchain.llms import OpenAI


class CountingCache(BaseCache):
    """A cache wrapper that counts cache hits and misses."""

    def __init__(self):
        """Initialize an empty cache and zeroed counters."""
        self._cache = {}
        self.hits = 0
        self.misses = 0

    # LangChain 0.0.x interface
    def lookup(self, prompt: str, llm_string: str):
        """Retrieve a cached result for older LangChain versions."""
        key = (prompt, llm_string)
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        return None

    def update(self, prompt: str, llm_string: str, return_val):
        """Store a result for older LangChain versions."""
        key = (prompt, llm_string)
        self._cache[key] = return_val

    # LangChain 0.1+ interface
    def get(self, key: str):
        """Retrieve a cached result for newer LangChain versions."""
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        return None

    def set(self, key: str, value):
        """Store a result for newer LangChain versions."""
        self._cache[key] = value


def print_cache_stats(cache: CountingCache):
    """Print cache hit and miss counts."""
    print(f"Cache hits: {cache.hits}")
    print(f"Cache misses: {cache.misses}")
    print(f"Total cache lookups: {cache.hits + cache.misses}")


def main():
    """Run cached LLM calls and print cache statistics."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set the OPENAI_API_KEY environment variable.")
        return

    # Set up the LLM. The API key is read from the environment.
    llm = OpenAI(
        model_name="text-davinci-003",
        openai_api_key=api_key,
        max_tokens=50,
    )

    # Create a counting cache and set it as the global LLM cache.
    counting_cache = CountingCache()
    set_llm_cache(counting_cache)

    # Run the same prompt multiple times to demonstrate caching.
    prompt = "What is the capital of France?"
    print("First call (should be a cache miss):")
    print(llm(prompt))
    print("\nSecond call (should be a cache hit):")
    print(llm(prompt))

    # Run a different prompt to show another miss.
    other_prompt = "What is the capital of Germany?"
    print("\nThird call (different prompt, should be a cache miss):")
    print(llm(other_prompt))

    # Print cache statistics after the cached LLM calls.
    print("\nCache statistics after cached LLM calls:")
    print_cache_stats(counting_cache)


if __name__ == "__main__":
    main()
