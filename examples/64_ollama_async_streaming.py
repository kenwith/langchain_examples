"""
Example: 64 - Ollama Async Streaming

This example demonstrates asynchronous streaming with Ollama using async generators.
It uses the provider-agnostic `init_chat_model` function to create an Ollama chat model,
then streams responses token by token with `astream`.

Requirements:
- Install `langchain`, `langchain-ollama`, and `ollama` (if not already installed).
- Have an Ollama server running locally (default: http://localhost:11434).
- Pull a model, e.g., `llama3.2`.

Environment variables (optional):
- `OLLAMA_BASE_URL`: Override the Ollama server URL.
- `OLLAMA_MODEL`: Model name to use (default: "llama3.2").
"""

import asyncio
import os

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage


def get_model():
    """Create an Ollama chat model using init_chat_model."""
    model_name = os.getenv("OLLAMA_MODEL", "llama3.2")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    return init_chat_model(
        model=model_name,
        provider="ollama",
        base_url=base_url,
        temperature=0.7,
    )


async def stream_response(prompt: str):
    """Stream a response from the Ollama model asynchronously."""
    model = get_model()
    messages = [HumanMessage(content=prompt)]
    async for chunk in model.astream(messages):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print()  # final newline


async def main():
    """Run the async streaming example."""
    prompt = "Explain the concept of async generators in Python in a few sentences."
    print(f"Prompt: {prompt}\n")
    print("Streaming response:")
    await stream_response(prompt)


if __name__ == "__main__":
    asyncio.run(main())
