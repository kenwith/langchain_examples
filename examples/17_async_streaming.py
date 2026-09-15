"""
| # | Example | Description |
|---|---------|-------------|
| 17 | Async Streaming | Concurrent token streaming from multiple model calls using async streams |

Stream tokens concurrently from multiple model calls using `astream` and
`asyncio.gather`.
"""

import asyncio
import os

from langchain.chat_models import init_chat_model


async def stream_one(label: str, model, prompt: str) -> str:
    """Stream one prompt and return the full response.

    Args:
        label: Identifier for this concurrent task.
        model: Any LangChain chat model that supports `astream`.
        prompt: User prompt to send.

    Returns:
        The full generated response text.
    """
    chunks: list[str] = []
    print(f"[{label}] Started: {prompt}")
    async for chunk in model.astream(prompt):
        if isinstance(chunk.content, str):
            chunks.append(chunk.content)
            print(f"[{label}] {chunk.content}", flush=True)
    print(f"[{label}] Done.")
    return "".join(chunks)


async def stream_concurrently(model, prompts: list[str]) -> list[str]:
    """Run multiple streaming calls concurrently.

    Args:
        model: Any LangChain chat model that supports `astream`.
        prompts: List of prompts to send.

    Returns:
        List of responses, ordered to match `prompts`.
    """
    labels = [f"task-{i + 1}" for i in range(len(prompts))]
    return await asyncio.gather(
        *(stream_one(label, model, prompt) for label, prompt in zip(labels, prompts))
    )


def main() -> None:
    """Run the concurrent async streaming demo."""
    model = init_chat_model(
        os.getenv("MODEL", "gpt-4o-mini"),
        model_provider=os.getenv("MODEL_PROVIDER"),
        temperature=0.7,
    )

    prompts = [
        "Tell me a one-sentence fun fact about the ocean.",
        "Tell me a one-sentence fun fact about space.",
        "Tell me a one-sentence fun fact about computers.",
    ]

    responses = asyncio.run(stream_concurrently(model, prompts))

    print("\n=== Full responses ===")
    for prompt, response in zip(prompts, responses):
        print(f"{prompt}\n{response}\n")


if __name__ == "__main__":
    main()
