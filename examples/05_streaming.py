"""Stream responses from chat models with a custom callback handler, with fallback for providers that do not support streaming, and optional text-file output.

Streaming options:
- Use `model.stream()` to stream tokens as they are generated.
- Pass a custom callback handler (e.g., `StreamingCallbackHandler`) to process tokens in real time.
- Set `streaming=True` on the model constructor to enable streaming for `invoke()` as well.
- Token usage is extracted from the final chunk's `usage_metadata` when available.
- Use `stream_to_console()` for a minimal token-by-token console streaming example.
- Use `stream_response()` to yield tokens for custom processing.
- Use `stream_to_file()` to write streamed tokens directly to a text file.
- Use `stream_with_events()` with `astream_events` for token streaming, the final stop reason, and event metadata.

Streaming configuration:
- `model.stream()` is the recommended streaming API and works without extra constructor options.
- Some providers (e.g., OpenAI) also support streaming for `invoke()` when `streaming=True` is set on the model constructor.
- Token usage may appear in individual chunks via `usage_metadata`; this example prints it as soon as it is found in a chunk.
"""

import asyncio

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


def print_chunk_usage_if_present(chunk):
    """Print token usage from a chunk if it contains usage_metadata."""
    usage = getattr(chunk, "usage_metadata", None)
    if usage:
        print("\n", flush=True)
        print_token_usage(usage)


def stream_to_console(model, messages):
    """Stream tokens directly to the console as they are generated.

    This is the simplest way to see token-by-token streaming. It uses
    `model.stream()` and prints each content chunk immediately.
    Falls back to a normal response if the provider does not support streaming.
    """
    if not hasattr(model, "stream"):
        print("The selected provider does not support streaming; falling back to normal response.\n")
        response = model.invoke(messages)
        print(response.content)
        print_token_usage(getattr(response, "usage_metadata", None))
        return

    print("Streaming response directly to console:\n", flush=True)
    chunks = []
    try:
        for chunk in model.stream(messages):
            content = getattr(chunk, "content", "")
            if content:
                print(content, end="", flush=True)
            chunks.append(chunk)
            print_chunk_usage_if_present(chunk)
        print("\n", flush=True)

        usage = None
        for chunk in reversed(chunks):
            usage = getattr(chunk, "usage_metadata", None)
            if usage:
                break
        if not any(getattr(c, "usage_metadata", None) for c in chunks):
            print_token_usage(usage)
    except NotImplementedError:
        print("\nThe selected provider does not support streaming; falling back to normal response.\n")
        response = model.invoke(messages)
        print(response.content)
        print_token_usage(getattr(response, "usage_metadata", None))


def stream_response(model, messages):
    """Yield response tokens as they are generated.

    This generator can be used for custom processing of each token. It falls
    back to yielding the full response content if the provider does not support
    streaming.
    """
    if not hasattr(model, "stream"):
        response = model.invoke(messages)
        content = response.content
        if isinstance(content, str):
            yield content
        else:
            yield str(content)
        return

    try:
        for chunk in model.stream(messages):
            content = getattr(chunk, "content", "")
            if content:
                yield content
    except NotImplementedError:
        response = model.invoke(messages)
        content = response.content
        if isinstance(content, str):
            yield content
        else:
            yield str(content)


def stream_to_file(model, messages, filepath):
    """Stream response tokens to a text file.

    Writes tokens to `filepath` as they are generated. Falls back to a normal
    response if the provider does not support streaming.
    """
    if not hasattr(model, "stream"):
        print("The selected provider does not support streaming; falling back to normal response.\n")
        response = model.invoke(messages)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(response.content)
        print(f"Response saved to {filepath}")
        print_token_usage(getattr(response, "usage_metadata", None))
        return

    print(f"Streaming response to {filepath}:\n", flush=True)
    chunks = []
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            for chunk in model.stream(messages):
                content = getattr(chunk, "content", "")
                if content:
                    f.write(content)
                    f.flush()
                chunks.append(chunk)
                print_chunk_usage_if_present(chunk)
        print(f"\nStreamed response saved to {filepath}", flush=True)

        usage = None
        for chunk in reversed(chunks):
            usage = getattr(chunk, "usage_metadata", None)
            if usage:
                break
        if not any(getattr(c, "usage_metadata", None) for c in chunks):
            print_token_usage(usage)
    except NotImplementedError:
        print("\nThe selected provider does not support streaming; falling back to normal response.\n")
        response = model.invoke(messages)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(response.content)
        print(f"Response saved to {filepath}")
        print_token_usage(getattr(response, "usage_metadata", None))


def stream_or_fallback(model, messages):
    """Stream from a model using a custom callback handler, falling back to a normal response if streaming is unsupported."""
    if not hasattr(model, "stream"):
        print("The selected provider does not support streaming; falling back to normal response.\n")
        response = model.invoke(messages)
        print(response.content)
        print_token_usage(getattr(response, "usage_metadata", None))
        return

    try:
        print("Streaming response:\n", flush=True)
        handler = StreamingCallbackHandler()
        chunks = []
        for chunk in model.stream(messages, callbacks=[handler]):
            chunks.append(chunk)
            print_chunk_usage_if_present(chunk)

        print("\n", flush=True)

        usage = None
        for chunk in reversed(chunks):
            usage = getattr(chunk, "usage_metadata", None)
            if usage:
                break
        if not any(getattr(c, "usage_metadata", None) for c in chunks):
            print_token_usage(usage)

    except NotImplementedError:
        print("\nThe selected provider does not support streaming; falling back to normal response.\n")
        response = model.invoke(messages)
        print(response.content)
        print_token_usage(getattr(response, "usage_metadata", None))


async def stream_with_events(model, messages):
    """Stream tokens using astream_events, printing token-by-token chunks and the final stop reason."""
    print("Streaming response with astream_events:\n", flush=True)
    chunks = []
    event_metadata = {}
    usage_metadata = None
    stop_reason = None

    try:
        async for event in model.astream_events(messages, version="v1"):
            event_name = event.get("event")
            if event_name == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk is None:
                    continue
                content = getattr(chunk, "content", "")
                if content:
                    print(content, end="", flush=True)
                chunks.append(chunk)
                print_chunk_usage_if_present(chunk)
                if not event_metadata:
                    event_metadata = event.get("metadata", {})
                # Capture stop reason from final chunk metadata when available
                chunk_metadata = getattr(chunk, "response_metadata", {}) or {}
                reason = chunk_metadata.get("finish_reason") or chunk_metadata.get("stop_reason")
                if reason:
                    stop_reason = reason
            elif event_name == "on_chat_model_end":
                output = event.get("data", {}).get("output")
                if output is not None:
                    usage_metadata = getattr(output, "usage_metadata", None)
                    if not event_metadata:
                        event_metadata = event.get("metadata", {})
                    # If stop reason wasn't found in chunks, try the output message
                    if stop_reason is None:
                        output_metadata = getattr(output, "response_metadata", {}) or {}
                        stop_reason = output_metadata.get("finish_reason") or output_metadata.get("stop_reason")
    except NotImplementedError:
        print("\nThe selected provider does not support astream_events; falling back to normal response.\n")
        response = await model.ainvoke(messages)
        print(response.content)
        print_token_usage(getattr(response, "usage_metadata", None))
        return

    print("\n", flush=True)

    if stop_reason:
        print(f"\nStop reason: {stop_reason}")
    else:
        print("\nStop reason: not available")

    if event_metadata:
        print("\nEvent metadata:")
        for key, value in event_metadata.items():
            print(f"  {key}: {value}")

    if usage_metadata is None:
        for chunk in reversed(chunks):
            usage_metadata = getattr(chunk, "usage_metadata", None)
            if usage_metadata:
                break
    if not any(getattr(c, "usage_metadata", None) for c in chunks):
        print_token_usage(usage_metadata)


def main():
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    messages = [HumanMessage(content="What is the capital of France?")]

    # Simplest token-by-token streaming directly to the console
    stream_to_console(model, messages)

    # Use the callback-handler streaming approach
    stream_or_fallback(model, messages)

    # Stream tokens to a text file
    stream_to_file(model, messages, "stream_output.txt")

    # Use astream_events for granular metadata handling
    asyncio.run(stream_with_events(model, messages))

    # Use the stream_response generator to yield tokens for custom processing
    print("Streaming response with stream_response():\n", flush=True)
    for token in stream_response(model, messages):
        print(token, end="", flush=True)
    print("\n", flush=True)


if __name__ == "__main__":
    main()
