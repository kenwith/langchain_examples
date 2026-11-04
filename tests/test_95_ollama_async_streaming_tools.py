"""Smoke tests for the Ollama async streaming tools example.

These tests verify that the example file compiles, imports correctly, and
defines at least one asynchronous function.

| Test File | Example Under Test | Description |
|-----------|--------------------|-------------|
| test_95_ollama_async_streaming_tools.py | ollama_async_streaming_tools.py | Import, compile, and async function smoke tests |
"""

import ast
import functools
import importlib.util
import inspect
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

_EXAMPLE_CANDIDATES = [
    REPO_ROOT / "examples" / "ollama_async_streaming_tools.py",
    REPO_ROOT / "ollama_async_streaming_tools.py",
]
EXAMPLE_PATH = next((p for p in _EXAMPLE_CANDIDATES if p.exists()), _EXAMPLE_CANDIDATES[0])


@functools.lru_cache(maxsize=1)
def _example_module():
    """Load and return the example module, cached for reuse."""
    if not EXAMPLE_PATH.exists():
        pytest.fail(f"Example file not found: {EXAMPLE_PATH}")

    spec = importlib.util.spec_from_file_location("ollama_async_streaming_tools", EXAMPLE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_example_file_exists():
    assert EXAMPLE_PATH.exists(), f"Expected example file at {EXAMPLE_PATH}"


def test_example_compiles():
    source = EXAMPLE_PATH.read_text(encoding="utf-8")
    ast.parse(source)


def test_example_imports():
    module = _example_module()
    assert module.__name__ == "ollama_async_streaming_tools"


def test_example_has_async_functions():
    module = _example_module()
    async_funcs = [
        name
        for name, obj in inspect.getmembers(module)
        if inspect.iscoroutinefunction(obj)
    ]
    assert async_funcs, "The example should define at least one async function"


def test_example_uses_init_chat_model():
    source = EXAMPLE_PATH.read_text(encoding="utf-8")
    assert "init_chat_model" in source, "The example should use provider-agnostic init_chat_model"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
