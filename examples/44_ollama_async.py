"""
Example 44: Async Streaming Chat with Ollama
============================================

| # | Example      | Description                          |
|---|--------------|--------------------------------------|
| 44| ollama_async | Async streaming chat with Ollama     |

This example demonstrates how to use LangChain's provider-agnostic
`init_chat_model` to stream responses asynchronously from an Ollama model.
It also shows how to run multiple chat calls concurrently using `asyncio.gather`
and print completion messages as each response finishes.
"""

import asyncio
import os

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage


def get_model():
    """Create an Ollama chat model using provider-agnostic init_chat_model."""
    return init_chat_model(
        model=os.getenv("OLLAMA_MODEL", "llama3.2"),
        model_provider="ollama",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )


async def stream_chat(model, prompt: str) -> None:
    """Send a prompt, stream the response, and print a completion message."""
    print(f"\n--- Prompt: {prompt} ---")
    async for chunk in model.astream([HumanMessage(content=prompt)]):
        print(chunk.content, end="", flush=True)
    print(f"\n[Completed: {prompt}]")


async def main() -> None:
    """Run multiple async streaming chat calls concurrently."""
    model = get_model()
    prompts = [
        os.getenv(
            "OLLAMA_PROMPT",
            "Explain the concept of async programming in one sentence.",
        ),
        "What is the capital of France?",
        "Give me a brief history of the internet.",
    ]

    # Create a task for each prompt and run them concurrently
    tasks = [stream_chat(model, prompt) for prompt in prompts]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
