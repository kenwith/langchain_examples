"""Structured output validation cases for the LangChain examples suite.

This module extends ``tests/test_13_structured_output.py`` with focused
cases that confirm the validation layer rejects malformed or incomplete
payloads before downstream code consumes them. The schema is exercised
directly so the assertions stay deterministic and provider-agnostic; a
single binding check confirms the same schema can be attached to a chat
model through ``init_chat_model``.

| Test | Scenario | Expected result |
|------|----------|-----------------|
| test_missing_required_field_raises | ``full_name`` omitted | ``ValidationError`` naming the field |
| test_multiple_missing_fields_reported | empty payload | every required field reported |
| test_wrong_type_raises | ``age`` supplied as text | ``ValidationError`` |
| test_empty_string_rejected | ``full_name`` is empty | ``ValidationError`` |
| test_out_of_range_age_rejected | ``age`` is negative | ``ValidationError`` |
| test_none_for_required_field_rejected | ``full_name`` is ``None`` | ``ValidationError`` |
| test_extra_field_rejected | unknown key supplied | ``ValidationError`` |
| test_optional_field_may_be_omitted | ``notes`` absent | record validates |
| test_valid_payload_round_trips | complete payload | record validates |
| test_with_structured_output_binds_schema | model plus schema | runnable returned |
"""

from __future__ import annotations

import os
from typing import Optional

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from langchain.chat_models import init_chat_model


class ContactRecord(BaseModel):
    """Minimal schema used to exercise structured output validation."""

    model_config = ConfigDict(extra="forbid")

    full_name: str = Field(min_length=1, description="Display name for the contact.")
    age: int = Field(ge=0, le=130, description="Age in whole years.")
    role: str = Field(min_length=1, description="Role or job title.")
    notes: Optional[str] = Field(default=None, description="Free-form notes.")


def _build_model():
    """Return a provider-agnostic chat model built from environment settings."""
    return init_chat_model(
        model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        model_provider=os.getenv("MODEL_PROVIDER", "openai"),
        temperature=0,
    )


def test_missing_required_field_raises():
    """A payload without ``full_name`` is rejected and the field is named."""
    with pytest.raises(ValidationError) as excinfo:
        ContactRecord.model_validate({"age": 30, "role": "engineer"})
    assert "full_name" in str(excinfo.value)


def test_multiple_missing_fields_reported():
    """An empty payload reports every required field in the error summary."""
    with pytest.raises(ValidationError) as excinfo:
        ContactRecord.model_validate({})
    message = str(excinfo.value)
    for field_name in ("full_name", "age", "role"):
        assert field_name in message


def test_wrong_type_raises():
    """A non-numeric ``age`` is rejected by type coercion."""
    with pytest.raises(ValidationError):
        ContactRecord.model_validate(
            {"full_name": "Sample Contact", "age": "not-a-number", "role": "analyst"}
        )


def test_empty_string_rejected():
    """An empty ``full_name`` violates the minimum length constraint."""
    with pytest.raises(ValidationError):
        ContactRecord.model_validate(
            {"full_name": "", "age": 20, "role": "analyst"}
        )


def test_out_of_range_age_rejected():
    """A negative ``age`` violates the lower bound constraint."""
    with pytest.raises(ValidationError):
        ContactRecord.model_validate(
            {"full_name": "Sample Contact", "age": -1, "role": "analyst"}
        )


def test_none_for_required_field_rejected():
    """``None`` is not accepted for a required string field."""
    with pytest.raises(ValidationError):
        ContactRecord.model_validate(
            {"full_name": None, "age": 20, "role": "analyst"}
        )


def test_extra_field_rejected():
    """An unknown key is rejected because the schema forbids extras."""
    with pytest.raises(ValidationError):
        ContactRecord.model_validate(
            {
                "full_name": "Sample Contact",
                "age": 20,
                "role": "analyst",
                "unexpected": "value",
            }
        )


def test_optional_field_may_be_omitted():
    """The optional ``notes`` field defaults to ``None`` when absent."""
    record = ContactRecord.model_validate(
        {"full_name": "Sample Contact", "age": 20, "role": "analyst"}
    )
    assert record.notes is None


def test_valid_payload_round_trips():
    """A complete payload validates without raising."""
    record = ContactRecord.model_validate(
        {
            "full_name": "Sample Contact",
            "age": 34,
            "role": "analyst",
            "notes": "Example note.",
        }
    )
    assert record.full_name == "Sample Contact"
    assert record.age == 34
    assert record.role == "analyst"
    assert record.notes == "Example note."


def test_with_structured_output_binds_schema():
    """The schema can be attached to a chat model built from environment settings."""
    if not os.getenv("MODEL_PROVIDER"):
        pytest.skip("MODEL_PROVIDER is not configured.")
    try:
        model = _build_model()
    except Exception as exc:  # pragma: no cover - depends on local credentials
        pytest.skip(f"Chat model could not be initialised: {exc}")
    runnable = model.with_structured_output(ContactRecord)
    assert runnable is not None


if __name__ == "__main__":
    sample = {"full_name": "Sample Contact", "age": 34, "role": "analyst"}
    print("Valid payload:", ContactRecord.model_validate(sample))

    try:
        ContactRecord.model_validate({"age": 34, "role": "analyst"})
    except ValidationError as exc:
        print("Invalid payload rejected:")
        print(exc)

    if os.getenv("MODEL_PROVIDER"):
        try:
            structured = _build_model().with_structured_output(ContactRecord)
            print("Structured runnable:", structured)
        except Exception as exc:  # pragma: no cover - depends on local credentials
            print("Model binding skipped:", exc)
    else:
        print("Model binding skipped: MODEL_PROVIDER is not configured.")
