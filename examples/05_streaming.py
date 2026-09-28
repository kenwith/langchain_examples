"""Example 05: Streaming with LangChain.

This example demonstrates how to stream token deltas from a chat model
using a generator function. The generator yields each piece of the response
as it arrives, allowing the caller to process the stream incrementally.
"""

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
    and the full response is yielded as a single delta.
    """
    if hasattr(llm, "stream"):
        for chunk in llm.stream(messages):
            content = getattr(chunk, "content", None)
            if content:
                yield content
    else:
        # Fallback for providers that do not support streaming.
        print("The selected provider does not support streaming; falling back to normal response")
        response = llm.invoke(messages)
        yield response.content


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

    # Consume the generator and print each delta as it arrives.
    for delta in stream_response(llm, messages):
        print(delta, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
