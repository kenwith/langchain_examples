"""Tests for example syntax.

This test module parses and compiles every Python file in the examples
directory using the ``ast`` module and the built-in ``compile`` function to
catch syntax errors without executing any API calls. It deliberately does not
import or invoke ``init_chat_model`` to avoid triggering network requests or
requiring provider credentials.

In addition to valid syntax, every example must contain a non-empty module
docstring, an ``if __name__ == '__main__':`` guard, and only absolute imports
(no relative imports or wildcard imports) so it can be safely imported and run
as a script. The test also verifies that every imported module can be resolved
in the current environment using ``importlib.util.find_spec``, which checks
importability without executing the example module.

The test suite also explicitly requires at least two Ollama example files and
at least two Ollama async LangGraph example files. All discovered examples,
including these, must satisfy the same syntax, docstring, and import rules.

Example files are discovered dynamically at test collection time using
``pytest_generate_tests``, so new examples are tested automatically without
maintaining a hardcoded list.

| Test | Description |
|------|-------------|
| test_example_syntax | Parse and compile every example file and verify it has a non-empty module docstring, a main guard, valid imports, and resolvable import targets. |
| test_examples_directory_has_python_files | Ensure the examples directory contains Python files to validate. |
| test_examples_directory_has_ollama_files | Ensure the examples directory contains at least two Ollama example files. |
| test_examples_directory_has_ollama_async_langgraph_files | Ensure the examples directory contains at least two Ollama async LangGraph example files. |
| test_ollama_async_langgraph_example_syntax | Parse and compile every Ollama async LangGraph example file and verify it has a non-empty module docstring, a main guard, valid imports, and resolvable import targets. |
"""

import ast
import importlib.util
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def _example_files():
    """Return a sorted list of Python files in the examples directory."""
    if not EXAMPLES_DIR.is_dir():
        return []
    return sorted(EXAMPLES_DIR.rglob("*.py"))


def _is_ollama_example(path):
    """Return True if the file name indicates an Ollama example."""
    return "ollama" in path.name.lower()


def _is_ollama_async_langgraph_example(path):
    """Return True if the file name indicates an Ollama async LangGraph example."""
    name = path.name.lower()
    return "ollama" in name and "async" in name and "langgraph" in name


def _has_docstring(tree):
    """Return True if the module has a non-empty docstring."""
    docstring = ast.get_docstring(tree)
    return bool(docstring and docstring.strip())


def _has_main_guard(tree):
    """Return True if the module has an ``if __name__ == '__main__':`` guard."""
    for node in tree.body:
        if isinstance(node, ast.If):
            test = node.test
            if isinstance(test, ast.Compare):
                left = test.left
                right = test.comparators[0] if test.comparators else None
                if (
                    isinstance(left, ast.Name)
                    and left.id == "__name__"
                    and isinstance(right, ast.Constant)
                    and right.value == "__main__"
                ):
                    return True
    return False


def _has_only_absolute_imports(tree):
    """Return True if all imports are absolute and no wildcard imports are used."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level != 0:
                return False
            for alias in node.names:
                if alias.name == "*":
                    return False
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "*":
                    return False
    return True


def _unresolvable_imports(tree):
    """Return a list of module names that cannot be resolved in the current environment."""
    unresolved = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name
                if importlib.util.find_spec(module_name) is None:
                    unresolved.append(module_name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                if importlib.util.find_spec(node.module) is None:
                    unresolved.append(node.module)
    return unresolved


def _validate_example_file(path):
    """Parse, compile, and validate a single example file."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    compile(source, str(path), "exec")

    assert _has_docstring(tree), f"{path} is missing a non-empty module docstring"
    assert _has_main_guard(tree), f"{path} is missing an if __name__ == '__main__': guard"
    assert _has_only_absolute_imports(tree), f"{path} contains relative or wildcard imports"

    unresolved = _unresolvable_imports(tree)
    assert not unresolved, f"{path} imports cannot be resolved: {', '.join(unresolved)}"


def pytest_generate_tests(metafunc):
    """Dynamically parametrize tests with discovered example files."""
    example_files = _example_files()
    ollama_async_langgraph_files = [
        path for path in example_files if _is_ollama_async_langgraph_example(path)
    ]

    if "example_path" in metafunc.fixturenames:
        if metafunc.function.__name__ == "test_ollama_async_langgraph_example_syntax":
            params = ollama_async_langgraph_files
        else:
            params = example_files
        metafunc.parametrize("example_path", params)


def test_example_syntax(example_path):
    """Verify that a single example file has valid syntax, docstring, main guard, and imports."""
    _validate_example_file(example_path)


def test_examples_directory_has_python_files():
    """Ensure the examples directory contains Python files to validate."""
    assert len(_example_files()) > 0, "No Python files found in the examples directory"


def test_examples_directory_has_ollama_files():
    """Ensure the examples directory contains at least two Ollama example files."""
    ollama_files = [path for path in _example_files() if _is_ollama_example(path)]
    assert len(ollama_files) >= 2, "Expected at least two Ollama example files"


def test_examples_directory_has_ollama_async_langgraph_files():
    """Ensure the examples directory contains at least two Ollama async LangGraph example files."""
    async_langgraph_files = [
        path for path in _example_files() if _is_ollama_async_langgraph_example(path)
    ]
    assert len(async_langgraph_files) >= 2, "Expected at least two Ollama async LangGraph example files"


def test_ollama_async_langgraph_example_syntax(example_path):
    """Verify that an Ollama async LangGraph example file meets all syntax and import requirements."""
    _validate_example_file(example_path)
