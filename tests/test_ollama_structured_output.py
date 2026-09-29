"""Tests for examples/42_ollama_structured_output.py."""

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pydantic

# Test: 42_ollama_structured_output
# ---------------------------------

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
EXAMPLE_PATH = EXAMPLES_DIR / "42_ollama_structured_output.py"


def _load_example_module():
    """Load the example module from file path."""
    spec = importlib.util.spec_from_file_location(
        "example_42_ollama_structured_output", EXAMPLE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _find_expected_schema(module):
    """Return the Pydantic model class defined in the example module."""
    schemas = [
        value
        for value in vars(module).values()
        if isinstance(value, type)
        and issubclass(value, pydantic.BaseModel)
        and value.__module__ == module.__name__
    ]
    assert len(schemas) == 1, f"Expected exactly one schema, found {len(schemas)}"
    return schemas[0]


def test_main_mocks_init_chat_model():
    """main() works with a mocked init_chat_model and uses the expected schema."""
    module = _load_example_module()
    expected_schema = _find_expected_schema(module)

    mock_model = MagicMock()
    mock_structured_model = MagicMock()
    mock_model.with_structured_output.return_value = mock_structured_model

    with patch.object(module, "init_chat_model", return_value=mock_model) as mock_init:
        module.main()

    mock_init.assert_called_once()
    mock_model.with_structured_output.assert_called_once()
    args, kwargs = mock_model.with_structured_output.call_args
    if args:
        actual_schema = args[0]
    else:
        actual_schema = kwargs.get("schema")
    assert actual_schema is expected_schema


if __name__ == "__main__":
    test_main_mocks_init_chat_model()
    print("All tests passed!")
