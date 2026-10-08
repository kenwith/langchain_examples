import pytest
from langchain.llms import FakeListLLM
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
from langchain_examples.cache import print_cache_stats


def test_cache_stats_output(capsys):
    """Verify that print_cache_stats() emits expected cache statistics."""
    set_llm_cache(InMemoryCache())
    llm = FakeListLLM(responses=["first", "second", "third"])

    # Populate the cache with a mix of hits and misses.
    llm.predict("hello")
    llm.predict("hello")
    llm.predict("world")

    print_cache_stats()

    captured = capsys.readouterr()
    assert "Cache hits:" in captured.out
    assert "Cache misses:" in captured.out
    assert "Cache size:" in captured.out
