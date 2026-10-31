"""Test async LLM caching."""

import asyncio
import os

from langchain.cache import InMemoryCache
from langchain.chat_models import init_chat_model
from langchain.globals import set_llm_cache


# ---------------------------------------------------------------------------
# | Test                     | Description                                              |
# |--------------------------|----------------------------------------------------------|
# | test_async_llm_caching   | Verifies async caching returns consistent results and    |
# |                          | exercises the cache.                                     |
# ---------------------------------------------------------------------------


def test_async_llm_caching():
    """Verify that async LLM caching returns consistent results and exercises the cache."""
    # Use a provider-agnostic model via environment variable.
    # Example: CHAT_MODEL="openai:gpt-4o" or CHAT_MODEL="anthropic:claude-3-5-sonnet-20240620"
    model = os.getenv("CHAT_MODEL", "openai:gpt-4o")
    llm = init_chat_model(model, temperature=0)

    cache = InMemoryCache()
    set_llm_cache(cache)

    prompt = "What is the capital of France?"

    # First call populates the cache
    result1 = asyncio.run(llm.ainvoke(prompt))
    # Second call should hit the cache
    result2 = asyncio.run(llm.ainvoke(prompt))

    assert result1 == result2, "Async LLM caching returned inconsistent results"
    assert cache.lookup(prompt, llm) is not None, "Cache was not populated after first call"

    # Clean up to avoid affecting other tests
    set_llm_cache(None)


if __name__ == "__main__":
    test_async_llm_caching()
    print("Async LLM caching test passed.")
