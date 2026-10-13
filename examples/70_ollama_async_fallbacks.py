"""Example 70: Async Fallback Chains with Ollama

This example demonstrates how to use asynchronous fallback chains with Ollama models.
It uses `init_chat_model` to create model instances and `asyncio` to run the chain.

The primary model is attempted first; if it fails, the fallback model is used.
This pattern is useful for improving reliability when a particular model is unavailable
or returns an error.

Prerequisites:
- Ollama is installed and running locally.
- The specified models are pulled (e.g., `llama3` and `mistral`).

Environment variables:
- `OLLAMA_BASE_URL`: Optional, defaults to `http://localhost:11434`.
- `OLLAMA_MODEL`: Primary model name, defaults to `llama3`.
- `OLLAMA_FALLBACK_MODEL`: Fallback model name, defaults to `mistral`.
"""

import asyncio
import os

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage


def create_model(model_name: str) -> object:
    """Create an Ollama chat model using init_chat_model."""
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    return init_chat_model(
        model=model_name,
        provider="ollama",
        base_url=base_url,
    )


async def run_async_fallback() -> None:
    """Run an async chain with a fallback model."""
    primary_model_name = os.getenv("OLLAMA_MODEL", "llama3")
    fallback_model_name = os.getenv("OLLAMA_FALLBACK_MODEL", "mistral")

    primary_model = create_model(primary_model_name)
    fallback_model = create_model(fallback_model_name)

    # Build a chain that falls back to the second model if the first fails.
    chain = primary_model.with_fallbacks([fallback_model])

    message = HumanMessage(content="Explain the concept of async fallbacks in one sentence.")
    response = await chain.ainvoke([message])

    print(f"Primary model: {primary_model_name}")
    print(f"Fallback model: {fallback_model_name}")
    print(f"Response: {response.content}")


async def main() -> None:
    """Main async entry point."""
    await run_async_fallback()


if __name__ == "__main__":
    asyncio.run(main())
