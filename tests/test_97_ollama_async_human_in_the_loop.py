"""Smoke test for the ollama_async_human_in_the_loop example.

Verifies that the example module imports successfully and defines the
expected asynchronous graph function.
"""

import asyncio
import importlib.util
import inspect
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Test: ollama_async_human_in_the_loop
# ---------------------------------------------------------------------------
# | Attribute   | Expected Type        |
# |-------------|----------------------|
# | async_graph | CoroutineFunction    |
# ---------------------------------------------------------------------------


EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "97_ollama_async_human_in_the_loop.py"
)


def _load_example_module():
    """Load the example module from the examples directory."""
    if not EXAMPLE_PATH.is_file():
        raise FileNotFoundError(f"Example module not found: {EXAMPLE_PATH}")

    spec = importlib.util.spec_from_file_location(
        "ollama_async_human_in_the_loop", EXAMPLE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_example_defines_async_graph_function():
    """Verify the example defines an async function named ``async_graph``."""
    module = _load_example_module()

    assert hasattr(module, "async_graph"), (
        "Expected the example to define an 'async_graph' function."
    )

    async_graph = getattr(module, "async_graph")
    assert inspect.iscoroutinefunction(async_graph), (
        "'async_graph' must be an async function."
    )


def _demo_async_graph():
    """Run a short async graph demo using a provider-agnostic chat model."""
    from langchain.chat_models import init_chat_model

    model = init_chat_model(os.getenv("OLLAMA_MODEL", "ollama/llama3"))
    module = _load_example_module()
    print(f"Loaded model: {model}")
    print(f"Async graph function: {module.async_graph}")


if __name__ == "__main__":
    test_example_defines_async_graph_function()
    print("Smoke test passed.")

    if "--demo" in sys.argv:
        _demo_async_graph()
