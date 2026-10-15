"""
Test for the async_llm_as_judge module.

| Test | Description |
|------|-------------|
| test_run_callable | Verify the run function is callable. |
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain.chat_models import init_chat_model

import async_llm_as_judge


def test_run_callable():
    """Verify that async_llm_as_judge.run is callable."""
    assert callable(async_llm_as_judge.run)


def demo():
    """Initialize a chat model using provider-agnostic configuration."""
    model = init_chat_model(
        model=os.getenv("CHAT_MODEL", "gpt-4o"),
        model_provider=os.getenv("CHAT_MODEL_PROVIDER", "openai"),
        temperature=0,
    )
    return model


if __name__ == "__main__":
    test_run_callable()
    print("Smoke test passed: async_llm_as_judge.run is callable.")
