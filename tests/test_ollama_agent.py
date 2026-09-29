"""Tests for examples/43_ollama_agent.py.

These tests mock `init_chat_model` so the example can be imported and run
end-to-end without requiring a live Ollama server or any API credentials.
The tests are provider-agnostic: they verify the example’s integration logic
without depending on a specific model backend.
"""

from __future__ import annotations

import importlib.util
import runpy
from pathlib import Path
from unittest.mock import MagicMock, patch

# | Test | Description |
# |------|-------------|
# | test_example_imports | Verify the example module can be imported with a mocked init_chat_model. |
# | test_example_runs_end_to_end | Run the example as __main__ with a mocked init_chat_model and assert it completes. |


def _example_path() -> Path:
    """Return the absolute path to examples/43_ollama_agent.py."""
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "examples" / "43_ollama_agent.py"


def _mock_chat_model() -> MagicMock:
    """Create a mock chat model that supports the agent’s expected interface."""
    model = MagicMock()
    # The agent may call invoke() directly or via bind_tools().
    model.invoke.return_value = "Mock response"
    model.bind_tools.return_value = model
    return model


def test_example_imports() -> None:
    """The example module should import cleanly when init_chat_model is mocked."""
    with patch("langchain.chat_models.init_chat_model", return_value=_mock_chat_model()):
        example_path = _example_path()
        spec = importlib.util.spec_from_file_location("ollama_agent_example", example_path)
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        # The module should expose a callable entry point (main or run).
        assert callable(getattr(module, "main", None)) or callable(getattr(module, "run", None))


def test_example_runs_end_to_end() -> None:
    """Running the example as __main__ should succeed with a mocked model."""
    with patch("langchain.chat_models.init_chat_model", return_value=_mock_chat_model()):
        example_path = _example_path()
        # run_path executes the module’s __main__ demo block.
        runpy.run_path(str(example_path), run_name="__main__")


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
