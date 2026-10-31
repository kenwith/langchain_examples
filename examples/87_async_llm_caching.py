"""Example: Async LLM Caching with init_chat_model and asyncio.gather

This example demonstrates how to cache LLM calls when using the provider-agnostic
`init_chat_model` function. It runs multiple async calls concurrently with
`asyncio.gather` and shows the performance benefit of caching repeated prompts.

The example uses `langchain_core.caches.InMemoryCache` as the cache backend.
The model is selected via `init_chat_model` and requires an API key for the
chosen provider. The provider and model name can be configured with the
environment variables `CHAT_MODEL_PROVIDER` and `CHAT_MODEL_NAME`.

| Provider   | Model             | Env Variable       |
|------------|-------------------|--------------------|
| OpenAI     | gpt-4o-mini       | OPENAI_API_KEY     |
| Anthropic  | claude-3-5-sonnet | ANTHROPIC_API_KEY  |
| Google     | gemini-1.5-pro    | GOOGLE_API_KEY     |
| ...        | ...               | ...                |

Run the example with:
    python examples/87_async_llm_caching.py
"""

import asyncio
import os
import time

from langchain.chat_models import init_chat_model
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache


def get_chat_model():
    """Create a chat model using init_chat_model from environment configuration."""
    provider = os.getenv("CHAT_MODEL_PROVIDER", "openai")
    model = os.getenv("CHAT_MODEL_NAME", "gpt-4o-mini")
    return init_chat_model(model, model_provider=provider)


async def invoke_with_cache(model, prompt):
    """Invoke the model asynchronously and return the response content."""
    response = await model.ainvoke(prompt)
    return response.content


async def run_prompts(model, prompts):
    """Run multiple prompts concurrently and return the results."""
    tasks = [invoke_with_cache(model, prompt) for prompt in prompts]
    return await asyncio.gather(*tasks)


async def main():
    """Main async function demonstrating cached LLM calls."""
    # Set up a global in-memory cache
    set_llm_cache(InMemoryCache())

    # Create the chat model (requires API key for the selected provider)
    model = get_chat_model()

    prompts = [
        "Say hello in one word.",
        "Say goodbye in one word.",
        "Say thank you in one word.",
    ]

    print("First set of calls (cache misses):")
    start = time.perf_counter()
    results = await run_prompts(model, prompts)
    elapsed = time.perf_counter() - start
    for prompt, result in zip(prompts, results):
        print(f"  Prompt: {prompt!r} -> {result!r}")
    print(f"  Time: {elapsed:.2f}s")

    print("\nSecond set of calls (cache hits):")
    start = time.perf_counter()
    results = await run_prompts(model, prompts)
    elapsed = time.perf_counter() - start
    for prompt, result in zip(prompts, results):
        print(f"  Prompt: {prompt!r} -> {result!r}")
    print(f"  Time: {elapsed:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
