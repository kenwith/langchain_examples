"""
Example 10: Async/Await

| # | Example | Description |
|---|---------|-------------|
| 10 | async | Demonstrate concurrent API calls with async/await |

This example shows how to use LangChain's async API to run multiple chat model calls concurrently.

Async patterns:
- `async def` defines a coroutine that can pause at `await` points.
- `await chain.arun(...)` yields control to the event loop while the
  model request is in flight, allowing other tasks to run.
- `asyncio.gather` schedules multiple coroutines concurrently and waits for
  all of them to finish.
- `chain.astream(...)` returns an async iterator that yields chunks as they
  are generated, enabling streaming responses.

Concurrency limits:
- `asyncio.gather` starts all tasks at once. With many prompts, this can
  overwhelm the API or hit rate limits. In production, use an
  `asyncio.Semaphore` to cap the number of concurrent requests.
"""

import asyncio
import os

from langchain.chains import LLMChain
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate


def get_model():
    """Initialize a chat model using environment variables."""
    model_name = os.getenv("MODEL", "gpt-4o-mini")
    model_provider = os.getenv("MODEL_PROVIDER", "openai")
    return init_chat_model(model_name, model_provider=model_provider)


def get_chain():
    """Initialize a chat model and wrap it in a simple prompt chain."""
    model = get_model()
    prompt = ChatPromptTemplate.from_template("Answer this question: {input}")
    return LLMChain(prompt=prompt, llm=model)


async def ask_model(chain, prompt: str) -> str:
    """Send a single prompt to the chain and return the response text.

    The `await` inside `chain.arun` lets the event loop run other tasks while
    the network request is in progress.
    """
    try:
        response = await chain.arun(input=prompt)
        return response.strip()
    except Exception as e:
        # Log the error and re-raise so the caller can decide how to handle it.
        print(f"Error asking model for prompt '{prompt}': {e}")
        raise


async def main() -> None:
    """Run multiple prompts concurrently and print their responses.

    Also demonstrates streaming with `astream`.
    """
    prompts = [
        "What is LangChain?",
        "What is an async function?",
        "What is the capital of France?",
    ]

    chain = get_chain()
    tasks = [ask_model(chain, prompt) for prompt in prompts]

    # Concurrency limit note: asyncio.gather fires all tasks at once. If you
    # have many prompts, consider wrapping ask_model in an asyncio.Semaphore
    # to limit concurrent API calls and avoid rate limits.
    try:
        responses = await asyncio.gather(*tasks)
    except Exception as e:
        print(f"One or more concurrent model calls failed: {e}")
        return

    for prompt, response in zip(prompts, responses):
        print(f"Prompt: {prompt}\nResponse: {response}\n")

    # Demonstrate streaming with astream
    print("Streaming response for 'Explain async/await in one sentence':")
    async for chunk in chain.astream({"input": "Explain async/await in one sentence"}):
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
