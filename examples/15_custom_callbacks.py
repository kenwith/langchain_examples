"""Example showing how to use a BaseCallbackHandler to record token usage and stream events."""

# ---
# title: Custom Callbacks
# description: Use a BaseCallbackHandler to record token usage and stream events.
# ---

import os

from langchain.chat_models import init_chat_model
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage


class TokenUsageCallbackHandler(BaseCallbackHandler):
    """Callback handler that records streamed tokens and usage metadata."""

    def __init__(self) -> None:
        self.streamed_tokens: list[str] = []
        self.usage_metadata: dict | None = None

    def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs) -> None:
        """Reset state when a new LLM call starts."""
        self.streamed_tokens = []
        self.usage_metadata = None

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Collect each token as it is streamed."""
        self.streamed_tokens.append(token)

    def on_llm_end(self, response, **kwargs) -> None:
        """Capture usage metadata from the final response."""
        if hasattr(response, "generations") and response.generations:
            generation = response.generations[0][0]
            self.usage_metadata = getattr(generation, "usage_metadata", None)

    def on_llm_error(self, error: Exception, **kwargs) -> None:
        """Report errors instead of failing silently."""
        print(f"LLM error: {error}")


def main() -> None:
    """Run a streaming chat example and display callback-recorded data."""
    model = init_chat_model(os.getenv("MODEL", "gpt-4o-mini"))
    handler = TokenUsageCallbackHandler()

    print("Streaming response:\n")
    response_text = ""
    for chunk in model.stream(
        [HumanMessage(content="Tell me a short joke about Python.")],
        callbacks=[handler],
    ):
        print(chunk.content, end="", flush=True)
        response_text += chunk.content
    print("\n")

    print("Callback handler data:")
    print(f"  Streamed tokens: {handler.streamed_tokens}")
    print(f"  Usage metadata: {handler.usage_metadata}")


if __name__ == "__main__":
    main()
