"""Example 05: Streaming output.

Use a StreamingStdOutCallbackHandler and print token deltas so the example shows
incremental output.
"""

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import OpenAI


def main() -> None:
    llm = OpenAI(
        model="gpt-3.5-turbo-instruct",
        temperature=0.7,
        max_tokens=256,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )

    prompt = "Write a short poem about streaming output."
    print("Prompt:")
    print(prompt)
    print("\nStreaming response:")
    llm(prompt)
    print()


if __name__ == "__main__":
    main()
