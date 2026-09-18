"""Unit tests for the async structured output example.

This test suite verifies that the async structured output example returns
valid Pydantic objects. The example is located at
``examples/async_structured_output.py`` and uses the provider-agnostic
``init_chat_model`` helper to create a chat model with a structured output
schema.

Test overview:

| Test function | Description |
| --- | --- |
| test_async_structured_output_returns_pydantic | Ensures the returned object is a Pydantic model. |
| test_async_structured_output_schema | Verifies the structured output uses the expected Pydantic schema. |
"""

import asyncio
import sys

import pytest

examples_async_structured_output = pytest.importorskip(
    "examples.async_structured_output",
    reason="Example module not found",
)

Person = getattr(examples_async_structured_output, "Person", None)
get_person = getattr(examples_async_structured_output, "get_person", None)

pytestmark = pytest.mark.skipif(
    Person is None or get_person is None,
    reason="Example must define Person and get_person",
)


class FakeChatModel:
    """A fake chat model that mimics structured output for testing."""

    def __init__(self):
        self.schema = None

    def with_structured_output(self, schema):
        """Store the schema and return the same instance."""
        self.schema = schema
        return self

    async def ainvoke(self, prompt):
        """Return a Pydantic instance based on the configured schema."""
        return self.schema(name="John Doe", age=30)


def test_async_structured_output_returns_pydantic(monkeypatch):
    """The example must return a valid Pydantic object."""
    fake_model = FakeChatModel()
    monkeypatch.setattr(
        examples_async_structured_output,
        "init_chat_model",
        lambda *args, **kwargs: fake_model,
    )

    result = asyncio.run(get_person())

    assert isinstance(result, Person)
    assert result.name == "John Doe"
    assert result.age == 30


def test_async_structured_output_schema(monkeypatch):
    """The example must use the Person schema for structured output."""
    fake_model = FakeChatModel()
    monkeypatch.setattr(
        examples_async_structured_output,
        "init_chat_model",
        lambda *args, **kwargs: fake_model,
    )

    asyncio.run(get_person())

    assert fake_model.schema is Person


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
