"""Tests for example 62: Ollama async parallel tool calls.

+-------+-------------------------------------+------------------------------------------+
| Test  | Example                             | Description                              |
+-------+-------------------------------------+------------------------------------------+
| 62    | ollama_async_parallel_tool_calls    | Verifies the example runs end to end.    |
+-------+-------------------------------------+------------------------------------------+
"""

import asyncio
import importlib.util
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _ollama_available() -> bool:
    """Return True when the Ollama server is reachable."""
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    url = f"{host.rstrip('/')}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError):
        return False


def _load_example():
    """Load the example module, or return None if it cannot be imported."""
    candidates = [
        REPO_ROOT / "examples" / "ollama_async_parallel_tool_calls.py",
        REPO_ROOT / "ollama_async_parallel_tool_calls.py",
    ]
    for path in candidates:
        if path.exists():
            spec = importlib.util.spec_from_file_location(
                "ollama_async_parallel_tool_calls", path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    return None


def _run_example(example):
    """Run the example's main or run function, handling async callables."""
    runner = getattr(example, "main", None) or getattr(example, "run", None)
    if runner is None:
        pytest.fail("Example module does not define main() or run()")
    if asyncio.iscoroutinefunction(runner):
        asyncio.run(runner())
    else:
        runner()


def test_ollama_async_parallel_tool_calls():
    """Run the example end to end when Ollama is available."""
    if not _ollama_available():
        pytest.skip("Ollama server is not available")
    example = _load_example()
    if example is None:
        pytest.skip("Example module could not be imported")
    _run_example(example)


if __name__ == "__main__":
    test_ollama_async_parallel_tool_calls()
