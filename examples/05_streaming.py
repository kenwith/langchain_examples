"""Example 05: Streaming output.

Use a StreamingStdOutCallbackHandler for synchronous streaming and an
async helper using astream_events for asynchronous token streaming.
Both print token deltas so the example shows incremental output.

Note: Streaming support varies by provider. This example uses OpenAI's
legacy LLM interface. For other providers, adapt the callback handler or
event names to match the provider's streaming implementation.
"""

import asyncio

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import OpenAI


async def stream_tokens_async(llm: OpenAI, prompt: str) -> None:
    """Stream tokens asynchronously using astream_events."""
    async for event in llm.astream_events(prompt, version="v1"):
        # The on_llm_new_token event fires for each token produced by the model.
        # Printing the chunk incrementally displays the response in real time.
        if event["event"] == "on_llm_new_token":
            print(event["data"]["chunk"], end="", flush=True)
    print()


def main() -> None:
    llm = OpenAI(
        model="gpt-3.5-turbo-instruct",
        temperature=0.7,
        max_tokens=256,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )

    prompt = "Write a short poem about streaming output."
    print("Prompt:")
    print(prompt)

    print("\nSynchronous streaming response:")
    llm(prompt)
    print()

    async_llm = OpenAI(
        model="gpt-3.5-turbo-instruct",
        temperature=0.7,
        max_tokens=256,
        streaming=True,
    )
    print("\nAsynchronous streaming response:")
    asyncio.run(stream_tokens_async(async_llm, prompt))


if __name__ == "__main__":
    main()
