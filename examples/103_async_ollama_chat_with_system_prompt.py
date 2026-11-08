# ---
# title: Async Ollama Chat with System Prompt
# description: Use an Ollama chat model with a system prompt asynchronously.
# provider: Ollama
# async: true
# ---

"""Async example using an Ollama chat model with a system prompt.

This example demonstrates how to use the provider-agnostic ``init_chat_model``
function to create an Ollama chat model and invoke it asynchronously with a
system prompt. The model is configured with a system message that instructs it
to respond concisely, and a user question is sent for completion.

Ollama must be running locally with the specified model available. The model
name can be overridden with the ``OLLAMA_MODEL`` environment variable.
"""

import asyncio
import os

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage


async def run_ollama_chat_with_system_prompt() -> None:
    """Run an asynchronous Ollama chat completion with a system prompt."""
    model_name = os.getenv("OLLAMA_MODEL", "llama3.1")
    model = init_chat_model(model=model_name, model_provider="ollama")

    messages = [
        SystemMessage(content="You are a helpful assistant that answers concisely."),
        HumanMessage(content="What is the capital of France?"),
    ]

    response = await model.ainvoke(messages)
    print(response.content)


if __name__ == "__main__":
    asyncio.run(run_ollama_chat_with_system_prompt())
