"""Example 05: Streaming with LangChain.

This example demonstrates how to stream token deltas from a chat model
using a generator function. The generator yields each piece of the response
as it arrives, allowing the caller to process the stream incrementally.
It also includes a wrapper generator that flushes each token to the terminal
so the streaming is visible immediately.
"""

import sys
from collections.abc import Iterator

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI


def stream_response(
    llm: BaseLanguageModel,
    messages: list[BaseMessage],
) -> Iterator[str]:
    """Yield token deltas from the model's response.

    Args:
        llm: The language model to use.
        messages: The chat messages to send.

    Yields:
        The next token delta (a string) from the model's response.

    If the model does not support streaming, a fallback message is printed
    to stderr and the full response is yielded as a single delta.
    """
    if hasattr(llm, "stream"):
        for chunk in llm.stream(messages):
            # ChatGenerationChunk wraps the actual message in a `.message`
            # attribute, while some models may yield the message directly.
            message = getattr(chunk, "message", chunk)
            content = getattr(message, "content", None)
            if content:
                yield content
    else:
        # Fallback for providers that do not support streaming.
        print(
            "The selected provider does not support streaming; "
            "falling back to normal response",
            file=sys.stderr,
        )
        response = llm.invoke(messages)
        yield response.content


def stream_tokens(
    llm: BaseLanguageModel,
    messages: list[BaseMessage],
) -> Iterator[str]:
    """Yield token deltas and flush them to standard output.

    This generator wraps :func:`stream_response` and writes each token to
    stdout immediately, followed by a flush. This makes streaming visible
    in the terminal while still yielding the tokens for any caller that
    wants to process them.

    Args:
        llm: The language model to use.
        messages: The chat messages to send.

    Yields:
        The next token delta (a string) from the model's response.
    """
    for token in stream_response(llm, messages):
        sys.stdout.write(token)
        sys.stdout.flush()
        yield token


def main() -> None:
    """Run the streaming example."""
    # Create a chat model with streaming enabled.
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    messages = [
        HumanMessage(content="Write a short poem about streaming data."),
    ]

    print("Prompt:")
    for message in messages:
        print(f"  {message.content}")
    print("\nStreaming response:")

    # The stream_tokens generator writes and flushes each token to stdout;
    # we only need to add the final newline after the stream ends.
    for _ in stream_tokens(llm, messages):
        pass
    print()


if __name__ == "__main__":
    main()
