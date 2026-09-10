"""Unit tests for example 09: structured output schema parsing.

These tests verify that the schema parsing logic works correctly when the model
returns different output types (JSON string, dict, invalid data). The model is
mocked to avoid any real API calls.
"""

# ---------------------------------------------------------------------------
# Test: Structured Output Schema Parsing
# | Test | Description |
# |------|-------------|
# | test_parse_json_string | Verifies parsing of JSON string output |
# | test_parse_dict | Verifies parsing of dict output |
# | test_invalid_json_raises | Verifies error on invalid JSON string |
# | test_missing_field_raises | Verifies error on missing required field |
# | test_unsupported_type_raises | Verifies error on unsupported output type |
# ---------------------------------------------------------------------------

import pytest
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from unittest.mock import Mock, patch


class Person(BaseModel):
    """A simple person schema for structured output tests."""

    name: str = Field(description="The person's name")
    age: int = Field(description="The person's age")


def parse_structured_output(output, schema):
    """Parse a model response into a Pydantic schema.

    Args:
        output: The raw model output (str or dict).
        schema: The Pydantic schema to parse into.

    Returns:
        An instance of the schema.

    Raises:
        ValueError: If the output cannot be parsed or is unsupported.
    """
    if isinstance(output, str):
        try:
            return schema.model_validate_json(output)
        except Exception as exc:
            raise ValueError(f"Failed to parse JSON output: {exc}") from exc
    if isinstance(output, dict):
        try:
            return schema.model_validate(output)
        except Exception as exc:
            raise ValueError(f"Failed to parse dict output: {exc}") from exc
    raise ValueError(f"Unsupported output type: {type(output)}")


def get_structured_output(model, prompt, schema):
    """Invoke a model and parse the response into a schema.

    Args:
        model: The chat model to call.
        prompt: The prompt to send.
        schema: The Pydantic schema to parse into.

    Returns:
        An instance of the schema.
    """
    response = model.invoke(prompt)
    return parse_structured_output(response, schema)


def test_parse_json_string():
    """Verify that a JSON string response is parsed into a Person."""
    with patch("langchain.chat_models.init_chat_model") as mock_init:
        mock_model = Mock()
        mock_model.invoke.return_value = '{"name": "Alice", "age": 30}'
        mock_init.return_value = mock_model

        model = init_chat_model("fake-model", provider="fake")
        person = get_structured_output(model, "Extract Alice", Person)

        assert isinstance(person, Person)
        assert person.name == "Alice"
        assert person.age == 30


def test_parse_dict():
    """Verify that a dict response is parsed into a Person."""
    with patch("langchain.chat_models.init_chat_model") as mock_init:
        mock_model = Mock()
        mock_model.invoke.return_value = {"name": "Bob", "age": 25}
        mock_init.return_value = mock_model

        model = init_chat_model("fake-model", provider="fake")
        person = get_structured_output(model, "Extract Bob", Person)

        assert isinstance(person, Person)
        assert person.name == "Bob"
        assert person.age == 25


def test_invalid_json_raises():
    """Verify that invalid JSON raises a ValueError."""
    with patch("langchain.chat_models.init_chat_model") as mock_init:
        mock_model = Mock()
        mock_model.invoke.return_value = "not json"
        mock_init.return_value = mock_model

        model = init_chat_model("fake-model", provider="fake")

        with pytest.raises(ValueError, match="Failed to parse JSON output"):
            get_structured_output(model, "Extract", Person)


def test_missing_field_raises():
    """Verify that a missing required field raises a ValueError."""
    with patch("langchain.chat_models.init_chat_model") as mock_init:
        mock_model = Mock()
        mock_model.invoke.return_value = '{"name": "Alice"}'
        mock_init.return_value = mock_model

        model = init_chat_model("fake-model", provider="fake")

        with pytest.raises(ValueError, match="Failed to parse JSON output"):
            get_structured_output(model, "Extract Alice", Person)


def test_unsupported_type_raises():
    """Verify that an unsupported output type raises a ValueError."""
    with patch("langchain.chat_models.init_chat_model") as mock_init:
        mock_model
