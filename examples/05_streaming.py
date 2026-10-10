"""Streaming example for LangChain.

Streaming is a technique where the model emits output tokens incrementally
rather than waiting for the full completion. This reduces perceived latency,
improves interactivity, and is essential for chat and real-time applications.

This example demonstrates how to stream tokens from a language model
using a helper function that yields token deltas and handles streaming
errors gracefully. It also includes a simple callback handler that counts
tokens during streaming to show how custom event handling works with LangChain.

Streaming usage notes:
- Use model.stream(...) to obtain a streaming response.
- Pass the response to stream_to_stdout() to print tokens as they arrive.
- The stream_response() generator yields token deltas and catches errors,
  so a failed stream produces an error message instead of crashing.
- Callback handlers can be passed to model.stream() to observe events
  such as on_llm_new_token.

To run this script:
1. Set the OPENAI_API_KEY environment variable to your OpenAI API key.
2. Install dependencies: pip install langchain-openai
3. Run the script: python examples/05_streaming.py
"""

import os
import time

from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI


class TokenCounterHandler(BaseCallbackHandler):
    """Callback handler that counts tokens during streaming."""

    def __init__(self):
        self.token_count = 0

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Increment the token counter for each new token."""
        self.token_count += 1


def print_token_with_delay(token: str, delay: float = 0.05) -> None:
    """Print a token and pause briefly to make streaming visible.

    Args:
        token: The token text to print.
        delay: Seconds to wait after printing the token.
    """
    print(token, end="", flush=True)
    time.sleep(delay)


def stream_response(response):
    """Yield token deltas from a streaming response.

    This generator yields each piece of content as it arrives. If an error
    occurs during streaming, it yields a descriptive error message and then
    stops gracefully instead of raising an exception.

    Args:
        response: An iterable of token chunks (e.g., from model.stream()).

    Yields:
        str: The next token delta, or an error message if streaming fails.
    """
    try:
        for chunk in response:
            yield chunk.content
    except Exception as e:
        yield f"[Stream error: {e}]"
        return


def stream_to_stdout(response, delay: float = 0.05) -> None:
    """Stream tokens from a response to standard output.

    This helper consumes the token deltas yielded by :func:`stream_response`
    and prints them as they arrive, using :func:`print_token_with_delay` to
    make the streaming visible. It also ensures the output ends with a newline
    so subsequent messages start on a fresh line.

    Args:
        response: An iterable of token chunks (e.g., from model.stream()).
        delay: Seconds to wait after printing each token.
    """
    ended_with_newline = False
    for token in stream_response(response):
        print_token_with_delay(token, delay)
        ended_with_newline = token.endswith("\n")
    if not ended_with_newline:
        print()


def main():
    """Run a simple streaming example with a token-counting callback."""
    # Use environment variables for credentials—never hardcode keys.
    model = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        streaming=True,
    )

    # Create a callback handler to count tokens as they stream.
    token_handler = TokenCounterHandler()

    # Create a streaming response by calling stream() on the model.
    # Pass the callback handler to receive streaming events.
    response = model.stream(
        "Write a short poem about streaming.",
        callbacks=[token_handler],
    )

    # Stream the tokens to stdout using the helper.
    stream_to_stdout(response)

    print("[Stream complete]", flush=True)

    # Display the token count collected by the callback handler.
    print(f"Token count: {token_handler.token_count}")


if __name__ == "__main__":
    main()
