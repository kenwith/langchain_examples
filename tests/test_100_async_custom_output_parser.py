"""
Tests for the async custom output parser example.

This module verifies the behavior of the custom async output parser
implemented in the example, including successful parsing, error handling,
and integration with a chat model.
"""

import asyncio
import os
from typing import Any, List

import pytest
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.messages import AIMessage

# Import the example module (assumed to be in examples directory)
from examples._100_async_custom_output_parser import AsyncCustomOutputParser, parse_with_model


class TestAsyncCustomOutputParser:
    """Test suite for the async custom output parser."""

    def test_parse_async_success(self):
        """Test that the async parse method returns expected output."""
        parser = AsyncCustomOutputParser()
        text = "item1\nitem2\nitem3"
        expected = ["item1", "item2", "item3"]
        result = asyncio.run(parser.parse_async(text))
        assert result == expected

    def test_parse_async_empty_input(self):
        """Test that empty input returns an empty list."""
        parser = AsyncCustomOutputParser()
        result = asyncio.run(parser.parse_async(""))
        assert result == []

    def test_parse_async_invalid_format(self):
        """Test that malformed input raises an error."""
        parser = AsyncCustomOutputParser()
        with pytest.raises(ValueError, match="Invalid format"):
            asyncio.run(parser.parse_async("no newlines here"))

    def test_parse_with_model(self, monkeypatch):
        """Test the full pipeline using a mocked chat model."""
        # Mock the chat model to return a fixed response
        class MockModel:
            async def ainvoke(self, messages):
                return AIMessage(content="apple\nbanana\ncherry")

        # Replace the model in the example's function
        monkeypatch.setattr(
            "examples._100_async_custom_output_parser.init_chat_model",
            lambda *args, **kwargs: MockModel(),
        )

        result = asyncio.run(parse_with_model("fruit list"))
        assert result == ["apple", "banana", "cherry"]


if __name__ == "__main__":
    # Demonstrate the parser usage
    async def demo():
        parser = AsyncCustomOutputParser()
        text = "one\ntwo\nthree"
        print("Parsing:", text)
        result = await parser.parse_async(text)
        print("Result:", result)

    asyncio.run(demo())
