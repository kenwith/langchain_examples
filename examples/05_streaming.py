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
  - The `on_llm_end` event fires when the model finishes, and includes
    token usage information in the output.
  - The `on_tool_start` event fires when a tool begins execution.
  - The `on_tool_end` event fires after a tool completes, with the tool output.
  - The `on_tool_error` event fires if a tool raises an exception.

Token usage information:
  - For synchronous streaming, a callback handler captures the usage from
    the `on_llm_end` callback.
  - For asynchronous streaming, the `on_llm_end` event provides the usage
    in its output.
  - Usage details include prompt tokens, completion tokens, and total tokens.

Error handling:
  - Network or API errors during streaming are caught and reported with a
    clear message instead of an unhandled traceback.

Note: Streaming support varies by provider. This example uses OpenAI's
legacy LLM interface. For other providers, adapt the stream method or
event names to match the provider's streaming implementation.
"""

import asyncio
from typing import Any

from langchain.callbacks.base import BaseCallbackHandler
from langchain.llms import OpenAI


class TokenUsageCallbackHandler(BaseCallbackHandler):
    """Capture token usage from the LLM on completion."""

    def __init__(self) -> None:
        self.token_usage: dict | None = None

    def on_llm_end(self, response, **kwargs) -> None:
        """Store token usage from the LLM result."""
        if response is not None and hasattr(response, "llm_output"):
            self.token_usage = response.llm_output.get("token_usage")


def print_token_usage(token_usage: dict | None) -> None:
    """Print token usage information in a readable format."""
    if not token_usage:
        print("[Token usage] Not available")
        return

    prompt_tokens = token_usage.get("prompt_tokens", "N/A")
    completion_tokens = token_usage.get("completion_tokens", "N/A")
    total_tokens = token_usage.get("total_tokens", "N/A")
    print(
        f"[Token usage] prompt_tokens={prompt_tokens}, "
        f"completion_tokens={completion_tokens}, total_tokens={total_tokens}"
    )


async def stream_response(llm: Any, prompt: str) -> None:
    """Stream tokens and tool events from an LLM or chain.

    Uses `astream_events` to capture token-level output, tool lifecycle
    events, and token usage on completion. For simple token streaming,
    `astream` can be used instead.

    Token streaming behavior:
      - The `on_llm_new_token` event fires for each token produced by the model.
      - Printing the chunk incrementally displays the response in real time.

    Token usage behavior:
      - The `on_llm_end` event fires when the model finishes and contains
        token usage in its output.

    Tool streaming behavior:
      - The `on_tool_start` event fires when a tool begins execution.
      - The `on_tool_end` event fires after a tool completes, with the tool output.
      - The `on_tool_error` event fires if a tool raises an exception.

    Error handling:
      - Network or API errors during streaming are caught and reported with a
        clear message instead of an unhandled traceback.
    """
    got_llm_end = False
    try:
        async for event in llm.astream_events(prompt, version="v1"):
            event_type = event["event"]
            if event_type == "on_llm_new_token":
                print(event["data"]["chunk"], end="", flush=True)
            elif event_type == "on_llm_end":
                got_llm_end = True
                output = event["data"]["output"]
                if hasattr(output, "llm_output"):
                    token_usage = output.llm_output.get("token_usage")
                elif isinstance(output, dict):
                    token_usage = output.get("llm_output", {}).get("token_usage")
                else:
                    token_usage = None
                print()
                print_token_usage(token_usage)
            elif event_type == "on_tool_start":
                print(f"\n[Tool start] {event['name']}")
            elif event_type == "on_tool_end":
                print(f"\n[Tool end] {event['name']}: {event['data']['output']}")
            elif event_type == "on_tool_error":
                print(f"\n[Tool error] {event['name']}: {event['data']['error']}")
    except Exception as exc:
        print(f"\n[Streaming error] A network or API error occurred: {exc}")
        return

    if not got_llm_end:
        print("\n[Token usage] Not available")


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
    usage_handler = TokenUsageCallbackHandler()
    llm.callbacks = [usage_handler]
    try:
        for chunk in llm.stream(prompt):
            print(chunk, end="", flush=True)
    except Exception as exc:
        print(f"\n[Streaming error] A network or API error occurred: {exc}")
    else:
        print()
        print_token_usage(usage_handler.token_usage)

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
