"""Streaming example for LangChain.

This example demonstrates how to stream tokens from a language model
using a helper function that prints each token as it arrives.
"""

import os

from langchain_openai import ChatOpenAI


def stream_response(response):
    """Print tokens from a streaming response as they arrive.

    Args:
        response: An iterable of token chunks (e.g., from model.stream()).
    """
    for chunk in response:
        print(chunk.content, end="", flush=True)
    print()  # Ensure a newline after the stream ends
    print("[Stream complete]", flush=True)


def main():
    """Run a simple streaming example."""
    # Use environment variables for credentials—never hardcode keys.
    model = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        streaming=True,
    )

    # Create a streaming response by calling stream() on the model.
    response = model.stream("Write a short poem about streaming.")

    # Use our helper to print tokens as they arrive.
    stream_response(response)


if __name__ == "__main__":
    main()
