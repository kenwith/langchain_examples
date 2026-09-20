"""Tests for the basic_chains module."""

from unittest.mock import patch

from langchain_core.messages import AIMessage

from basic_chains import build_chain


class FakeChatModel:
    """A fake chat model that returns a canned response."""

    def __init__(self, *args, **kwargs):
        """Accept and ignore any constructor arguments."""
        pass

    def invoke(self, messages):
        """Return a fixed AIMessage."""
        return AIMessage(content="Hello from the fake model!")


def test_build_chain_uses_patched_model():
    """Test that build_chain uses the patched model and does not hit the network."""
    with patch("basic_chains.ChatOpenAI", FakeChatModel):
        chain = build_chain()
        result = chain.invoke({"topic": "unit testing"})
        assert "Hello from the fake model!" in str(result)


def test_build_chain_with_empty_topic():
    """Test that build_chain handles an empty topic without errors."""
    with patch("basic_chains.ChatOpenAI", FakeChatModel):
        chain = build_chain()
        result = chain.invoke({"topic": ""})
        assert "Hello from the fake model!" in str(result)


def test_build_chain_with_special_characters():
    """Test that build_chain handles special characters in the topic."""
    special_topic = "!@#$%^&*()_+{}|:<>?~`-=[]\\;',./\n\t héllo wörld 你好 🎉"
    with patch("basic_chains.ChatOpenAI", FakeChatModel):
        chain = build_chain()
        result = chain.invoke({"topic": special_topic})
        assert "Hello from the fake model!" in str(result)
