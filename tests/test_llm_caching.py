"""Tests for LLM caching using fake models and InMemoryCache.

| Test | Description |
|------|-------------|
| test_repeated_calls_are_cached | Ensures the same prompt returns a cached response and the model is called only once. |
| test_different_prompts_are_not_cached | Ensures different prompts bypass the cache and invoke the model each time. |
| test_cache_clear_forces_new_call | Ensures clearing the cache causes the model to be called again. |
"""

import os

from langchain.chat_models import init_chat_model
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult


class CountingFakeModel(FakeMessagesListChatModel):
    """A fake chat model that counts how many times it generates a response."""

    def __init__(self, response: str = "Hello!"):
        super().__init__(responses=[AIMessage(content=response)])
        self.call_count = 0

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.call_count += 1
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def _make_cached_model() -> CountingFakeModel:
    """Create a fake model with a fresh InMemoryCache."""
    model = CountingFakeModel()
    set_llm_cache(InMemoryCache())
    return model


def test_repeated_calls_are_cached():
    model = _make_cached_model()
    try:
        first = model.invoke("Hello")
        second = model.invoke("Hello")
        assert first.content == "Hello!"
        assert second.content == "Hello!"
        assert model.call_count == 1
    finally:
        set_llm_cache(None)


def test_different_prompts_are_not_cached():
    model = _make_cached_model()
    try:
        model.invoke("Hello")
        model.invoke("World")
        assert model.call_count == 2
    finally:
        set_llm_cache(None)


def test_cache_clear_forces_new_call():
    model = _make_cached_model()
    try:
        model.invoke("Hello")
        set_llm_cache(InMemoryCache())
        model.invoke("Hello")
        assert model.call_count == 2
    finally:
        set_llm_cache(None)


if __name__ == "__main__":
    test_repeated_calls_are_cached()
    test_different_prompts_are_not_cached()
    test_cache_clear_forces_new_call()
    print("All tests passed!")

    # Provider-agnostic init_chat_model demo with caching.
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        model = init_chat_model("gpt-4o", model_provider="openai", api_key=api_key)
        set_llm_cache(InMemoryCache())
        response1 = model.invoke("Say hello")
        response2 = model.invoke("Say hello")
        print(f"Cached response: {response2.content}")
        set_llm_cache(None)
    else:
        print("Set OPENAI_API_KEY to run the init_chat_model demo.")
