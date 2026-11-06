"""Example 05: Streaming output.

Use the `stream` method for synchronous streaming and a `stream_response`
helper using `astream_events` for asynchronous token and tool streaming.
Both print token deltas so the example shows incremental output.

Token streaming behavior:
  - The `stream` method yields each token as it is produced by the model.
  - Printing each chunk incrementally displays the response in real time.

Async streaming behavior:
  - The `astream` method provides a simple async token stream.
  - The `astream_events` method provides a detailed event stream that
    includes token, tool, and chain events.
  - The `on_llm_new_token` event fires for each token produced by the model.
  - The `on_tool_start` event fires when a tool begins execution.
  - The `on_tool_end` event fires after a tool completes, with the tool output.
  - The `on_tool_error` event fires if a tool raises an exception.

Note: Streaming support varies by provider. This example uses OpenAI's
legacy LLM interface. For other providers, adapt the stream method or
event names to match the provider's streaming implementation.
"""

import asyncio
from typing import Any

from langchain.llms import OpenAI


async def stream_response(llm: Any, prompt: str) -> None:
    """Stream tokens and tool events from an LLM or chain.

    Uses `astream_events` to capture token-level output and tool lifecycle
    events. For simple token streaming, `astream` can be used instead.

    Token streaming behavior:
      - The `on_llm_new_token` event fires for each token produced by the model.
      - Printing the chunk incrementally displays the response in real time.

    Tool streaming behavior:
      - The `on_tool_start` event fires when a tool begins execution.
      - The `on_tool_end` event fires after a tool completes, with the tool output.
      - The `on_tool_error` event fires if a tool raises an exception.
    """
    async for event in llm.astream_events(prompt, version="v1"):
        event_type = event["event"]
        if event_type == "on_llm_new_token":
            print(event["data"]["chunk"], end="", flush=True)
        elif event_type == "on_tool_start":
            print(f"\n[Tool start] {event['name']}")
        elif event_type == "on_tool_end":
            print(f"\n[Tool end] {event['name']}: {event['data']['output']}")
        elif event_type == "on_tool_error":
            print(f"\n[Tool error] {event['name']}: {event['data']['error']}")
    print()


def main() -> None:
    llm = OpenAI(
        model="gpt-3.5-turbo-instruct",
        temperature=0.7,
        max_tokens=256,
        streaming=True,
    )

    prompt = "Write a short poem about streaming output."
    print("Prompt:")
    print(prompt)

    print("\nSynchronous streaming response:")
    for chunk in llm.stream(prompt):
        print(chunk, end="", flush=True)
    print()

    async_llm = OpenAI(
        model="gpt-3.5-turbo-instruct",
        temperature=0.7,
        max_tokens=256,
        streaming=True,
    )
    print("\nAsynchronous streaming response:")
    asyncio.run(stream_response(async_llm, prompt))


if __name__ == "__main__":
    main()
