"""Tests for example syntax.

This test module parses every Python file in the examples directory using
the ``ast`` module to catch syntax errors without executing any API calls.
It deliberately does not import or invoke ``init_chat_model`` to avoid
triggering network requests or requiring provider credentials.

| Test | Description |
|------|-------------|
| test_example_syntax | Parse every example file with ``ast`` to catch syntax errors. |
| test_examples_directory_has_python_files | Ensure the examples directory contains Python files to validate. |
"""

import ast
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def _example_files():
    """Return a sorted list of Python files in the examples directory."""
    if not EXAMPLES_DIR.is_dir():
        return []
    return sorted(EXAMPLES_DIR.rglob("*.py"))


def test_examples_directory_has_python_files():
    """Fail if there are no Python files in the examples directory."""
    files = _example_files()
    assert files, f"No Python files found in {EXAMPLES_DIR}"


@pytest.mark.parametrize(
    "example_path",
    _example_files(),
    ids=lambda p: p.relative_to(EXAMPLES_DIR).as_posix(),
)
def test_example_syntax(example_path):
    """Assert that the example file contains valid Python syntax."""
    source = example_path.read_text(encoding="utf-8")
    # Parse only; do not execute to avoid any API calls or side effects.
    ast.parse(source, filename=str(example_path))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
