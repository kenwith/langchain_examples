"""Stream responses from chat models with a custom callback handler, with fallback for providers that do not support streaming.

Streaming options:
- Use `model.stream()` to stream tokens as they are generated.
- Pass a custom callback handler (e.g., `StreamingCallbackHandler`) to process tokens in real time.
- Set `streaming=True` on the model constructor to enable streaming for `invoke()` as well.
- Token usage is extracted from the final chunk's `usage_metadata` when available.
"""

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


class StreamingCallbackHandler(BaseCallbackHandler):
    """Custom callback handler that prints tokens as they are generated."""

    def __init__(self):
        super().__init__()
        self.tokens = []

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Run on new token. Print it and store it."""
        self.tokens.append(token)
        print(token, end="", flush=True)


def print_token_usage(metadata):
    if not metadata:
        print("Token usage: not available")
        return

    if isinstance(metadata, dict):
        input_tokens = metadata.get("input_tokens") or metadata.get("prompt_tokens")
        output_tokens = metadata.get("output_tokens") or metadata.get("completion_tokens")
        total_tokens = metadata.get("total_tokens")
    else:
        input_tokens = getattr(metadata, "input_tokens", None)
        output_tokens = getattr(metadata, "output_tokens", None)
        total_tokens = getattr(metadata, "total_tokens", None)

    print("Token summary:")
    print(f"  Prompt tokens: {input_tokens}")
    print(f"  Completion tokens: {output_tokens}")
    print(f"  Total tokens: {total_tokens}")


def stream_or_fallback(model, messages):
    """Stream from a model using a custom callback handler, falling back to a normal response if streaming is unsupported."""
    if not hasattr(model, "stream"):
        print("Provider does not support streaming; falling back to normal response.\n")
        response = model.invoke(messages)
        print(response.content)
        print_token_usage(getattr(response, "usage_metadata", None))
        return

    try:
        print("Streaming response:\n")
        handler = StreamingCallbackHandler()
        chunks = []
        for chunk in model.stream(messages, callbacks=[handler]):
            chunks.append(chunk)

        print("\n")

        usage = None
        for chunk in reversed(chunks):
            usage = getattr(chunk, "usage_metadata", None)
            if usage:
                break

        print_token_usage(usage)

    except NotImplementedError:
        print("\nStreaming not supported; falling back to normal response.\n")
        response = model.invoke(messages)
        print(response.content)
        print_token_usage(getattr(response, "usage_metadata", None))


def main():
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    messages = [HumanMessage(content="What is the capital of France?")]
    stream_or_fallback(model, messages)


if __name__ == "__main__":
    main()
