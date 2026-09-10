"""
Example 10: Async/Await

| # | Example | Description |
|---|---------|-------------|
| 10 | async | Demonstrate concurrent API calls with async/await |

This example shows how to use LangChain's async API to run multiple chat model calls concurrently.
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
    """Send a single prompt to the model and return the response text."""
    response = await model.ainvoke(prompt)
    return response.content


async def run_concurrent(prompts: list[str]) -> list[str]:
    """Run multiple prompts concurrently and return all responses."""
    model = get_model()
    tasks = [ask_model(model, prompt) for prompt in prompts]
    return await asyncio.gather(*tasks)


def main() -> None:
    """Run the async example."""
    prompts = [
        "What is LangChain?",
        "What is an async function?",
        "What is the capital of France?",
    ]
    responses = asyncio.run(run_concurrent(prompts))
    for prompt, response in zip(prompts, responses):
        print(f"Prompt: {prompt}\nResponse: {response}\n")


if __name__ == "__main__":
    main()
