"""Tests for examples/60_ollama_custom_callbacks.py."""

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Paths
EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
EXAMPLE_PATH = EXAMPLES_DIR / "60_ollama_custom_callbacks.py"


def _load_example():
    """Load the example module, skipping if dependencies are missing."""
    try:
        spec = importlib.util.spec_from_file_location("example_60_ollama_custom_callbacks", EXAMPLE_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as exc:  # pragma: no cover - depends on environment
        pytest.skip(f"Could not load example module: {exc}")


@pytest.fixture(scope="module")
def example_module():
    """Fixture providing the loaded example module."""
    return _load_example()


# ---------------------------------------------------------------------------
# Test: Example structure
# ---------------------------------------------------------------------------


def test_example_has_custom_callback_handler(example_module):
    """The example should define a custom callback handler class."""
    assert hasattr(example_module, "CustomCallbackHandler")


def test_custom_callback_handler_implements_base_methods(example_module):
    """The custom handler should implement the standard callback methods."""
    handler = example_module.CustomCallbackHandler()
    run_mock = MagicMock()

    # These should not raise
    handler.on_llm_start(run_mock)
    handler.on_llm_new_token("token", run_mock)
    handler.on_llm_end(run_mock)
    handler.on_llm_error(Exception("test"), run_mock)


def test_custom_callback_handler_records_tokens(example_module):
    """The custom handler should accumulate tokens when on_llm_new_token is called."""
    handler = example_module.CustomCallbackHandler()
    run_mock = MagicMock()
    handler.on_llm_new_token("Hello", run_mock)
    handler.on_llm_new_token(" world", run_mock)

    assert hasattr(handler, "tokens")
    assert "".join(handler.tokens) == "Hello world"


# ---------------------------------------------------------------------------
# Test: Main function
# ---------------------------------------------------------------------------


def test_example_has_main_function(example_module):
    """The example should expose a main function."""
    assert callable(getattr(example_module, "main", None))


def test_main_uses_init_chat_model(example_module):
    """The main function should use init_chat_model to create the model."""
    if not hasattr(example_module, "main"):
        pytest.skip("Example does not have a main function")

    with patch.object(example_module, "init_chat_model") as mock_init:
        mock_model = MagicMock()
        mock_model.invoke.return_value = "Mock response"
        mock_init.return_value = mock_model

        example_module.main()

        mock_init.assert_called_once()
        mock_model.invoke.assert_called_once()


def test_main_passes_callbacks_to_model(example_module):
    """The main function should pass a callback handler to the model."""
    if not hasattr(example_module, "main"):
        pytest.skip("Example does not have a main function")

    with patch.object(example_module, "init_chat_model") as mock_init:
        mock_model = MagicMock()
        mock_init.return_value = mock_model

        example_module.main()

        # Verify that callbacks were supplied when creating the model
        _, kwargs = mock_init.call_args
        assert "callbacks" in kwargs
        callbacks = kwargs["callbacks"]
        assert len(callbacks) == 1
        assert isinstance(callbacks[0], example_module.CustomCallbackHandler)


# ---------------------------------------------------------------------------
# Test: Example runs end-to-end with mocked model
# ---------------------------------------------------------------------------


def test_main_returns_response(example_module):
    """The main function should return the model response."""
    if not hasattr(example_module, "main"):
        pytest.skip("Example does not have a main function")

    with patch.object(example_module, "init_chat_model") as mock_init:
        mock_model = MagicMock()
        mock_model.invoke.return_value = "Hello from mock"
        mock_init.return_value = mock_model

        result = example_module.main()

        assert result == "Hello from mock"


# ---------------------------------------------------------------------------
# Test: __main__ demo block
# ---------------------------------------------------------------------------


def test_example_has_main_guard(example_module):
    """The example should include a __main__ guard."""
    source = EXAMPLE_PATH.read_text()
    assert 'if __name__ == "__main__":' in source
    assert "main()" in source
