"""Tests for verifying custom callback handlers record LLM lifecycle events.

This module follows the repository conventions: provider-agnostic model
initialization, plain test functions, README-style section headers, and a
small ``__main__`` demo.
"""

import os

from langchain.chat_models import init_chat_model
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage


class CallbackRecorder(BaseCallbackHandler):
    """Custom callback handler that records LLM start and end events."""

    def __init__(self):
        self.starts = []
        self.ends = []

    def on_llm_start(self, serialized, prompts, **kwargs):
        """Record the beginning of an LLM call."""
        self.starts.append({"prompts": prompts, "kwargs": kwargs})

    def on_llm_end(self, response, **kwargs):
        """Record the end of an LLM call."""
        self.ends.append({"response": response, "kwargs": kwargs})


def test_custom_callbacks_record_start_and_end():
    """Verify that the callback handler records both on_llm_start and on_llm_end.

    ## Expected Behavior
    - One start event is recorded for a single LLM invocation.
    - One end event is recorded for the same invocation.
    """
    callback_handler = CallbackRecorder()

    model = init_chat_model(
        os.getenv("LLM_MODEL", "gpt-4o-mini"),
        model_provider=os.getenv("LLM_PROVIDER", "openai"),
        api_key=os.getenv("LLM_API_KEY"),
        callbacks=[callback_handler],
    )

    model.invoke([HumanMessage(content="Say hello")])

    assert len(callback_handler.starts) == 1
    assert len(callback_handler.ends) == 1


if __name__ == "__main__":
    # Run manually to see the events printed (or use pytest).
    recorder = CallbackRecorder()
    chat_model = init_chat_model(
        os.getenv("LLM_MODEL", "gpt-4o-mini"),
        model_provider=os.getenv("LLM_PROVIDER", "openai"),
        api_key=os.getenv("LLM_API_KEY"),
        callbacks=[recorder],
    )
    chat_model.invoke("Hello from the demo block!")
    print(f"Start events recorded: {len(recorder.starts)}")
    print(f"End events recorded: {len(recorder.ends)}")
