"""Example demonstrating custom callback handlers with an Ollama chat model.

This example uses init_chat_model to create an Ollama-backed chat model and
a custom BaseCallbackHandler to observe streaming events.
"""

import os

from langchain.chat_models import init_chat_model
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_core.outputs import LLMResult


class CustomCallbackHandler(BaseCallbackHandler):
    """A simple callback handler that logs chat model events."""

    def on_chat_model_start(self, serialized, messages, **kwargs):
        """Log when the chat model starts."""
        print("--- on_chat_model_start ---")
        print(f"Model: {serialized.get('name', 'unknown')}")
        for message_list in messages:
            for message in message_list:
                print(f"Message: {message.type} - {message.content}")

    def on_llm_new_token(self, token, **kwargs):
        """Log each token produced during streaming."""
        print(f"Token: {token!r}")

    def on_llm_end(self, response: LLMResult, **kwargs):
        """Log when the model finishes generating."""
        print("--- on_llm_end ---")
        for generation in response.generations:
            for gen in generation:
                print(f"Generated: {gen.text}")

    def on_llm_error(self, error: BaseException, **kwargs):
        """Log any errors raised by the model."""
        print(f"--- on_llm_error ---: {error}")


def run_demo():
    """Run the custom callback demo with an Ollama chat model."""
    model_name = os.getenv("OLLAMA_MODEL", "llama3.1")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    model = init_chat_model(
        f"ollama:{model_name}",
        temperature=0.1,
        base_url=base_url,
    )

    handler = CustomCallbackHandler()
    messages = [HumanMessage(content="Tell me a short joke about Python.")]

    print("Streaming response with custom callbacks...")
    for chunk in model.stream(messages, config={"callbacks": [handler]}):
        # The callback handler already prints tokens; this loop consumes the stream.
        pass

    print("\nDone.")


if __name__ == "__main__":
    run_demo()
