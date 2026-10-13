"""Stream tokens from a language model while collecting token count and elapsed time.

This example demonstrates two approaches: iterating over the stream directly,
and using a callback handler to receive tokens as they are generated.
"""

import os
import time

from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI


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

    for chunk in llm.stream(prompt):
        token = chunk.content
        if token:
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
    """Callback handler that collects tokens and timing during streaming."""

    def __init__(self):
        self.tokens = []
        self.token_count = 0
        self.start_time = None
        self.elapsed_time = None

    def on_llm_start(self, serialized, prompts, **kwargs):
        """Record the start time when generation begins."""
        self.start_time = time.perf_counter()

    def on_llm_new_token(self, token, **kwargs):
        """Collect and print each token as it is generated."""
        if token:
            self.tokens.append(token)
            self.token_count += 1
            print(token, end="", flush=True)

    def on_llm_end(self, response, **kwargs):
        """Record the elapsed time when generation completes."""
        self.elapsed_time = time.perf_counter() - self.start_time


def stream_with_callback_handler(llm, prompt):
    """Stream tokens using a callback handler, collecting token count and elapsed time.

    Args:
        llm: A LangChain chat model that supports streaming.
        prompt: The user prompt to send.

    Returns:
        The full response text.
    """
    handler = TokenCallbackHandler()
    response = llm.invoke(prompt, callbacks=[handler])

    elapsed_time = handler.elapsed_time or 0.0
    token_count = handler.token_count
    print("\n\n--- Callback Streaming Summary ---")
    print(f"Tokens received: {token_count}")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    if elapsed_time > 0:
        print(f"Throughput: {token_count / elapsed_time:.1f} tokens/second")

    return response.content


def main():
    """Run both streaming examples."""
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


if __name__ == "__main__":
    main()
