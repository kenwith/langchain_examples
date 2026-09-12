import pytest
from unittest.mock import Mock
from pydantic import BaseModel, ValidationError


class OutputModel(BaseModel):
    name: str
    age: int


def validate_output(data: dict) -> OutputModel:
    """Validate the output data, raising ValueError with a clear message."""
    try:
        return OutputModel(**data)
    except ValidationError as e:
        errors = "; ".join(f"{err['loc'][0]}: {err['msg']}" for err in e.errors())
        raise ValueError(f"Invalid output data: {errors}")


def process_row(row: dict, model_call) -> OutputModel:
    """Call the model and validate the result."""
    raw = model_call(row)
    return validate_output(raw)


def test_valid_row_passes():
    model_call = Mock(return_value={"name": "Alice", "age": 30})
    result = process_row({"id": 1}, model_call)
    assert result == OutputModel(name="Alice", age=30)


def test_invalid_row_raises_clear_error():
    model_call = Mock(return_value={"name": "Bob", "age": "not-an-int"})
    with pytest.raises(ValueError, match="age"):
        process_row({"id": 2}, model_call)


def test_missing_field_raises_error():
    model_call = Mock(return_value={"name": "Charlie"})
    with pytest.raises(ValueError, match="age"):
        process_row({"id": 3}, model_call)
