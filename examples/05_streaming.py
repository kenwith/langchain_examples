"""Handle providers that do not support streaming by falling back to a normal response and print a token summary."""

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


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
    if not hasattr(model, "stream"):
        print("Provider does not support streaming; falling back to normal response.\n")
        response = model.invoke(messages)
        print(response.content)
        print_token_usage(getattr(response, "usage_metadata", None))
        return

    try:
        print("Streaming response:\n")
        chunks = []
        for chunk in model.stream(messages):
            chunks.append(chunk)
            text = getattr(chunk, "content", "")
            if text:
                print(text, end="", flush=True)

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
