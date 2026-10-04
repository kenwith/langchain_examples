"""Streaming example for LangChain.

Streaming is a technique where the model emits output tokens incrementally
rather than waiting for the full completion. This reduces perceived latency,
improves interactivity, and is essential for chat and real-time applications.

This example demonstrates how to stream tokens from a language model
using a helper function that prints each token as it arrives.

To run this script:
1. Set the OPENAI_API_KEY environment variable to your OpenAI API key.
2. Install dependencies: pip install langchain-openai
3. Run the script: python examples/05_streaming.py
"""

import os

from langchain_openai import ChatOpenAI


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
