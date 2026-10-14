"""Tests for example syntax.

This test module parses and compiles every Python file in the examples
directory using the ``ast`` module and the built-in ``compile`` function to
catch syntax errors without executing any API calls. It deliberately does not
import or invoke ``init_chat_model`` to avoid triggering network requests or
requiring provider credentials.

In addition to valid syntax, every example must contain a non-empty module
docstring, an ``if __name__ == '__main__':`` guard, and only absolute imports
(no relative imports or wildcard imports) so it can be safely imported and run
as a script.

The test suite also explicitly requires at least two Ollama example files and
at least two Ollama async LangGraph example files. All discovered examples,
including these, must satisfy the same syntax, docstring, and import rules.

Example files are discovered dynamically at test collection time using
``pytest_generate_tests``, so new examples are tested automatically without
maintaining a hardcoded list.

| Test | Description |
|------|-------------|
| test_example_syntax | Parse and compile every example file and verify it has a non-empty module docstring, a main guard, and valid imports. |
| test_examples_directory_has_python_files | Ensure the examples directory contains Python files to validate. |
| test_examples_directory_has_ollama_files | Ensure the examples directory contains at least two Ollama example files. |
| test_examples_directory_has_ollama_async_langgraph_files | Ensure the examples directory contains at least two Ollama async LangGraph example files. |
| test_ollama_async_langgraph_example_syntax | Parse and compile every Ollama async LangGraph example file and verify it has a non-empty module docstring, a main guard, and valid imports. |
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


def _has_docstring(tree):
    """Return True if the module has a non-empty docstring."""
    docstring = ast.get_docstring(tree)
    return
