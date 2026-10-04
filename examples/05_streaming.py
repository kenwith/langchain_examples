"""Streaming example for LangChain.

Streaming is a technique where the model emits output tokens incrementally
rather than waiting for the full completion. This reduces perceived latency,
improves interactivity, and is essential for chat and real-time applications.

This example demonstrates how to stream tokens from a language model
using a helper function that prints each token as it arrives. It also
includes a simple callback handler that counts tokens during streaming
to show how custom event handling works with LangChain.

To run this script:
1. Set the OPENAI_API_KEY environment variable to your OpenAI API key.
2. Install dependencies: pip install langchain-openai
3. Run the script: python examples/05_streaming.py
"""

import os

from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI


class TokenCounterHandler(BaseCallbackHandler):
    """Callback handler that counts tokens during streaming."""

    def __init__(self):
        self.token_count = 0

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Increment the token counter for each new token."""
        self.token_count += 1


def stream_response(response):
    """Print tokens from a streaming response as they arrive.

    Ensures the stream output ends with exactly one newline before the
    completion message, avoiding extra blank lines when the model already
    emits a trailing newline.

    Args:
        response: An iterable of token chunks (e.g., from model.stream()).
    """
    ended_with_newline = False
    for chunk in response:
        content = chunk.content
        print(content, end="", flush=True)
        ended_with_newline = content.endswith("\n")

    if not ended_with_newline:
        print()  # Add a final newline if the stream didn't already end with one
    print("[Stream complete]", flush=True)


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

    # Use our helper to print tokens as they arrive.
    stream_response(response)

    # Display the token count collected by the callback handler.
    print(f"Token count: {token_handler.token_count}")


if __name__ == "__main__":
    main()
