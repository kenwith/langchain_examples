"""Tests for the Ollama async custom output parser example."""

import asyncio
import os
from typing import Any, Dict, List

import pytest
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel


class CustomAsyncOutputParser(BaseOutputParser[Dict[str, Any]]):
    """A custom output parser that extracts key-value pairs from text."""

    async def aparse(self, text: str) -> Dict[str, Any]:
        """Parse text into a dictionary of key-value pairs."""
        lines = text.strip().split("\n")
        result = {}
        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                result[key.strip()] = value.strip()
        return result

    def parse(self, text: str) -> Dict[str, Any]:
        """Synchronous parse (not used in async example)."""
        return asyncio.run(self.aparse(text))


def create_chain(model: Any) -> RunnableSequence:
    """Create a chain with a custom async output parser."""
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are a helpful assistant."), ("human", "{input}")]
    )
    parser = CustomAsyncOutputParser()
    return prompt | model | parser


@pytest.mark.asyncio
async def test_async_custom_output_parser_with_fake_model() -> None:
    """Test the custom async output parser with a fake model."""
    # Use a fake model that returns a predetermined response
    fake_messages = [AIMessage(content="key1: value1\nkey2: value2")]
    fake_model = FakeMessagesListChatModel(responses=fake_messages)

    chain = create_chain(fake_model)
    result = await chain.ainvoke({"input": "test"})

    assert result == {"key1": "value1", "key2": "value2"}


@pytest.mark.asyncio
async def test_async_custom_output_parser_with_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the custom async output parser with a mocked Ollama model."""
    # Mock the ChatOllama to avoid real API calls
    class MockOllama(ChatOllama):
        async def _agenerate(
            self,
            messages: List[Any],
            stop: List[str] | None = None,
            run_manager: Any = None,
            **kwargs: Any,
        ) -> ChatResult:
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="a: 1\nb: 2"))]
            )

    model = MockOllama(model="test-model", base_url="http://localhost:11434")
    chain = create_chain(model)
    result = await chain.ainvoke({"input": "test"})

    assert result == {"a": "1", "b": "2"}


def test_parse_method() -> None:
    """Test the synchronous parse method (for completeness)."""
    parser = CustomAsyncOutputParser()
    assert parser.parse("x: 1\ny: 2") == {"x": "1", "y": "2"}


if __name__ == "__main__":
    # Demo block: run the async test manually
    async def demo() -> None:
        """Run a small demo of the async custom output parser."""
        fake_messages = [AIMessage(content="foo: bar\nbaz: qux")]
        fake_model = FakeMessagesListChatModel(responses=fake_messages)
        chain = create_chain(fake_model)
        result = await chain.ainvoke({"input": "demo"})
        print(f"Parsed result: {result}")

    asyncio.run(demo())
