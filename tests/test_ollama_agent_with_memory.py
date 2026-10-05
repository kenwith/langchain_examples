"""Tests for the Ollama agent with memory example."""

import inspect
from unittest.mock import Mock

import pytest

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from examples.ollama_agent_with_memory import ask, create_agent


class FakeChatModel(BaseChatModel):
    """A minimal fake chat model that returns a fixed response."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="Hello from fake model"))]
        )

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"


class RecordingFakeModel(BaseChatModel):
    """A fake chat model that records the messages it receives."""

    def __init__(self):
        self.call_messages = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.call_messages.append(list(messages))
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content=f"Call {len(self.call_messages)}")
                )
            ]
        )

    @property
    def _llm_type(self) -> str:
        return "recording-fake-chat-model"


# ---------------------------------------------------------------------------
# Test: Provider-agnostic init_chat_model usage
# ---------------------------------------------------------------------------

def test_example_uses_init_chat_model():
    """The example module should call init_chat_model rather than a vendor model directly."""
    import examples.ollama_agent_with_memory as example_module

    source = inspect.getsource(example_module)
    assert "init_chat_model" in source


def test_init_chat_model_called_when_creating_agent(monkeypatch):
    """create_agent() should use init_chat_model to obtain the language model."""
    fake = FakeChatModel()
    mock_init = Mock(return_value=fake)
    monkeypatch.setattr(
        "examples.ollama_agent_with_memory.init_chat_model", mock_init
    )

    create_agent()

    mock_init.assert_called_once()


# ---------------------------------------------------------------------------
# Test: Agent response
# ---------------------------------------------------------------------------

def test_ask_returns_response(monkeypatch):
    """The agent should return a response when the model produces one."""
    fake = FakeChatModel()
    monkeypatch.setattr(
        "examples.ollama_agent_with_memory.get_llm", lambda: fake
    )

    result = ask("What is your name?")

    assert "Hello from fake model" in result


# ---------------------------------------------------------------------------
# Test: Memory retention
# ---------------------------------------------------------------------------

def test_agent_retains_memory_across_calls(monkeypatch):
    """Conversation history should be passed to the model on subsequent calls."""
    fake = RecordingFakeModel()
    monkeypatch.setattr(
        "examples.ollama_agent_with_memory.get_llm", lambda: fake
    )

    agent = create_agent()
    agent.invoke({"input": "First message"})
    agent.invoke({"input": "Second message"})

    assert len(fake.call_messages) == 2
    assert len(fake.call_messages[1]) > len(fake.call_messages[0])


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Demonstrate the example using a fake model, no Ollama server required.
    fake = FakeChatModel()
    import examples.ollama_agent_with_memory as example_module

    original_get_llm = example_module.get_llm
    example_module.get_llm = lambda: fake

    try:
        response = example_module.ask("Hello!")
        print("Demo response:", response)
    finally:
        example_module.get_llm = original_get_llm
