"""Stream tokens from a language model while collecting token count and elapsed time.

This example demonstrates three approaches to streaming token generation:

1. Direct iteration: the `stream` method yields chunks as they are produced,
   and a `stream_tokens()` helper encapsulates the token-by-token loop.
2. Callback handler: a callback object receives token events during generation.
3. Custom callback: a helper function creates a handler with a user-defined
   token callback for specialized processing (e.g., formatting or logging).

Each approach collects the full response, token count, and elapsed time, and
prints a summary of the streaming performance.
"""

import os
import time

from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI


def stream_tokens(llm, prompt):
    """Yield tokens from the LLM as they are generated.

    Args:
        llm: A LangChain chat model that supports streaming.
        prompt: The user prompt to send.

    Yields:
        str: Each token as it is generated.
    """
    for chunk in llm.stream(prompt):
        token = chunk.content
        if token:
            yield token


def stream_with_metrics(llm, prompt):
    """Stream tokens from the LLM, collecting token count and elapsed time.

    Args:
        llm: A LangChain chat model that supports streaming.
        prompt: The user prompt to send.

    Returns:
        The full response text.
    """
    token_count = 0
    start_time = time.perf_counter()
    chunks = []

    for token in stream_tokens(llm, prompt):
        chunks.append(token)
        token_count += 1
        print(token, end="", flush=True)

    elapsed_time = time.perf_counter() - start_time
    print("\n\n--- Streaming Summary ---")
    print(f"Tokens received: {token_count}")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    if elapsed_time > 0:
        print(f"Throughput: {token_count / elapsed_time:.1f} tokens/second")

    return "".join(chunks)


class TokenCallbackHandler(BaseCallbackHandler):
    """Callback handler that collects tokens and timing during streaming.

    Args:
        on_token: Optional callable invoked for each token. If provided, it
            receives the token string and can perform custom processing. The
            default behavior prints the token and collects it.
    """

    def __init__(self, on_token=None):
        self.tokens = []
        self.token_count = 0
        self.start_time = None
        self.elapsed_time = None
        self._on_token = on_token

    def on_llm_start(self, serialized, prompts, **kwargs):
        """Record the start time when generation begins."""
        self.start_time = time.perf_counter()

    def on_llm_new_token(self, token, **kwargs):
        """Collect and process each token as it is generated."""
        if token:
            self.tokens.append(token)
            self.token_count += 1
            if self._on_token:
                self._on_token(token)
            else:
                print(token, end="", flush=True)

    def on_llm_end(self, response, **kwargs):
        """Record the elapsed time when generation completes."""
        self.elapsed_time = time.perf_counter() - self.start_time


def create_token_callback_handler(on_token=None):
    """Create a TokenCallbackHandler with an optional custom token callback.

    Args:
        on_token: Optional callable that accepts a token string. When provided,
            it is called for each token instead of the default printing.

    Returns:
        A configured TokenCallbackHandler instance.
    """
    return TokenCallbackHandler(on_token=on_token)


def stream_with_callback_handler(llm, prompt):
    """Stream tokens using a callback handler, collecting token count and elapsed time.

    Args:
        llm: A LangChain chat model that supports streaming.
        prompt: The user prompt to send.

    Returns:
        The full response text.
    """
    handler = create_token_callback_handler()
    response = llm.invoke(prompt, callbacks=[handler])

    elapsed_time = handler.elapsed_time or 0.0
    token_count = handler.token_count
    print("\n\n--- Callback Streaming Summary ---")
    print(f"Tokens received: {token_count}")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    if elapsed_time > 0:
        print(f"Throughput: {token_count / elapsed_time:.1f} tokens/second")

    return response.content


def stream_with_custom_callback(llm, prompt):
    """Stream tokens using a custom callback that transforms each token.

    This example demonstrates how the helper function enables custom token
    processing while still collecting token count and elapsed time.

    Args:
        llm: A LangChain chat model that supports streaming.
        prompt: The user prompt to send.

    Returns:
        The full response text.
    """
    def uppercase_token(token):
        print(token.upper(), end="", flush=True)

    handler = create_token_callback_handler(on_token=uppercase_token)
    response = llm.invoke(prompt, callbacks=[handler])

    elapsed_time = handler.elapsed_time or 0.0
    token_count = handler.token_count
    print("\n\n--- Custom Callback Streaming Summary ---")
    print(f"Tokens received: {token_count}")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    if elapsed_time > 0:
        print(f"Throughput: {token_count / elapsed_time:.1f} tokens/second")

    return response.content


def main():
    """Run all three streaming examples."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")

    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, streaming=True)
    prompt = "Write a short poem about streaming data."

    print("=== Direct streaming ===")
    response = stream_with_metrics(llm, prompt)
    print(f"\nFull response:\n{response}")

    print("\n=== Callback handler streaming ===")
    response = stream_with_callback_handler(llm, prompt)
    print(f"\nFull response:\n{response}")

    print("\n=== Custom callback streaming ===")
    response = stream_with_custom_callback(llm, prompt)
    print(f"\nFull response:\n{response}")


if __name__ == "__main__":
    main()
