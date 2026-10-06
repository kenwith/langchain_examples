"""Tests for examples/55_ollama_fallbacks.py.

| Test | Description |
|------|-------------|
| test_get_model_returns_runnable_with_fallbacks | Verifies the returned model has a fallback configured. |
| test_get_model_uses_ollama_and_openai | Verifies the primary and fallback providers are configured correctly. |
| test_fallback_uses_openai_api_key_from_env | Verifies the fallback reads the API key from the environment. |
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest

EXAMPLE_PATH = Path(__file__).resolve().parents[1] / "examples" / "55_ollama_fallbacks.py"


@pytest.fixture(scope="module")
def example_module() -> ModuleType:
    """Load the example module from source."""
    spec = importlib.util.spec_from_file_location("ollama_fallbacks_example", EXAMPLE_PATH)
    module = importlib.util.module
