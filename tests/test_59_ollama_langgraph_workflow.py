"""Tests for the 59_ollama_langgraph_workflow example."""

import importlib.util
from pathlib import Path
from unittest.mock import Mock

import pytest

# ---------------------------------------------------------------------------
# Test: 59_ollama_langgraph_workflow
# ---------------------------------------------------------------------------


def load_example_module():
    """Load the example module from its file path."""
    example_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "59_ollama_langgraph_workflow.py"
    )
    spec = importlib.util.spec_from_file_location(
        "ollama_langgraph_workflow", example_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def example_module():
    """Provide the loaded example module for the test session."""
    return load_example_module()


class FakeChatModel:
    """Minimal fake chat model for testing without external services."""

    def __init__(self, *args, **kwargs):
        pass

    def invoke(self, messages, **kwargs):
        return "Mock response"

    def __call__(self, *args, **kwargs):
        return "Mock response"


@pytest.fixture
def fake_init_chat_model(monkeypatch, example_module):
    """Replace init_chat_model in the example module with a fake factory."""
    def _fake_init(*args, **kwargs):
        return FakeChatModel()

    monkeypatch.setattr(example_module, "init_chat_model", _fake_init)
    return _fake_init


def test_module_loads(example_module):
    """The example module should be importable without errors."""
    assert example_module is not None


def test_module_has_entry_point(example_module):
    """The example module should expose a callable entry point."""
    candidates = ["main", "run", "run_workflow", "workflow"]
    entry_point = next((name for name in candidates if hasattr(example_module, name)), None)
    assert entry_point is not None, f"Expected one of {candidates} in the example module"


def test_workflow_runs_with_mock_model(example_module, fake_init_chat_model):
    """The workflow should execute successfully when the model is mocked."""
    entry_point = next(
        name for name in ["main", "run", "run_workflow", "workflow"]
        if hasattr(example_module, name)
    )
    result = getattr(example_module, entry_point)()
    assert result is not None


def test_workflow_uses_init_chat_model(example_module, fake_init_chat_model):
    """The workflow should call init_chat_model at least once."""
    entry_point = next(
        name for name in ["main", "run", "run_workflow", "workflow"]
        if hasattr(example_module, name)
    )
    getattr(example_module, entry_point)()
    assert fake_init_chat_model.called


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
