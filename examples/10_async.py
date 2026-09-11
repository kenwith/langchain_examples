"""
Example 10: Async/Await

| # | Example | Description |
|---|---------|-------------|
| 10 | async | Demonstrate concurrent API calls with async/await |

This example shows how to use LangChain's async API to run multiple chat model calls concurrently.

Async patterns:
- `async def` defines a coroutine that can pause at `await` points.
- `await model.ainvoke(prompt)` yields control to the event loop while the
  model request is in flight, allowing other tasks to run.
- `asyncio.gather` schedules multiple coroutines concurrently and waits for
  all of them to finish.

Concurrency limits:
- `asyncio.gather` starts all tasks at once. With many prompts, this can
  overwhelm the API or hit rate limits. In production, use an
  `asyncio.Semaphore` to cap the number of concurrent requests.
"""

import asyncio
import os

from langchain.chat_models import init_chat_model


def get_model():
    """Initialize a chat model using environment variables."""
    model_name = os.getenv("MODEL", "gpt-4o-mini")
    model_provider = os.getenv("MODEL_PROVIDER", "openai")
    return init_chat_model(model_name, model_provider=model_provider)


async def ask_model(model, prompt: str) -> str:
    """Send a single prompt to the model and return the response text.

    The `await` inside `ainvoke` lets the event loop run other tasks while
    the network request is in progress.
    """
    try:
        response = await model.ainvoke(prompt)
        return response.content
    except Exception as e:
        # Log the error and re-raise so the caller can decide how to handle it.
        print(f"Error asking model for prompt '{prompt}': {e}")
        raise


async def run_concurrent(prompts: list[str]) -> list[str]:
    """Run multiple prompts concurrently and return all responses.

    This creates one coroutine per prompt and awaits them together with
    asyncio.gather. Because the model calls are I/O-bound, they run
    concurrently rather than sequentially.
    """
    model = get_model()
    tasks = [ask_model(model, prompt) for prompt in prompts]

    # Concurrency limit note: asyncio.gather fires all tasks at once. If you
    # have many prompts, consider wrapping ask_model in an asyncio.Semaphore
    # to limit concurrent API calls and avoid rate limits.
    try:
        return await asyncio.gather(*tasks)
    except Exception as e:
        print(f"One or more concurrent model calls failed: {e}")
        raise


def main() -> None:
    """Run the async example."""
    prompts = [
        "What is LangChain?",
        "What is an async function?",
        "What is the capital of France?",
    ]
    try:
        responses = asyncio.run(run_concurrent(prompts))
    except Exception as e:
        print(f"Async example failed: {e}")
        return

    for prompt, response in zip(prompts, responses):
        print(f"Prompt: {prompt}\nResponse: {response}\n")


if __name__ == "__main__":
    main()
