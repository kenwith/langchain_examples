"""Stream tokens from a language model while collecting token count and elapsed time."""

import os
import time

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


def main():
    """Run the streaming example."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")

    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, streaming=True)
    prompt = "Write a short poem about streaming data."

    response = stream_with_metrics(llm, prompt)
    print(f"\nFull response:\n{response}")


if __name__ == "__main__":
    main()
