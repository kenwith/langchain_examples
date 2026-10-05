"""Tests for example syntax.

This test module parses and compiles every Python file in the examples
directory using the ``ast`` module and the built-in ``compile`` function to
catch syntax errors without executing any API calls. It deliberately does not
import or invoke ``init_chat_model`` to avoid triggering network requests or
requiring provider credentials.

In addition to valid syntax, every example must contain a module docstring,
an ``if __name__ == '__main__':`` guard, and only absolute imports (no
relative imports or wildcard imports) so it can be safely imported and run
as a script.

The test suite also explicitly requires at least two Ollama example files
and verifies that they satisfy the same syntax, docstring, and import rules.

Example files are discovered dynamically at test collection time using
``pytest_generate_tests``, so new examples are tested automatically without
maintaining a hardcoded list.

| Test | Description |
|------|-------------|
| test_example_syntax | Parse and compile every example file and verify it has a module docstring, a main guard, and valid imports. |
| test_examples_directory_has_python_files | Ensure the examples directory contains Python files to validate. |
| test_examples_directory_has_ollama_files | Ensure the examples directory contains at least two Ollama example files. |
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
    """Return True if the module has a docstring."""
    return ast.get_docstring(tree) is not None


def _has_main_guard(tree):
    """Return True if the module has an ``if __name__ == '__main__':`` guard."""
    for node in tree.body:
        if isinstance(node, ast.If):
            test = node.test
            if (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == "__name__"
                and len(test.ops) == 1
                and isinstance(test.ops[0], ast.Eq)
                and len(test.comparators) == 1
                and isinstance(test.comparators[0], ast.Constant)
                and test.comparators[0].value == "__main__"
            ):
                return True
    return False


def _has_valid_imports(tree):
    """Return True if all imports are absolute and do not use wildcard imports."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level > 0:
                return False
            for alias in node.names:
                if alias.name == "*":
                    return False
    return True


def _is_ollama_example(path):
    """Return True if 'ollama' appears in the path or filename."""
    return "ollama" in path.as_posix().lower()


def test_examples_directory_has_python_files():
    """Fail if there are no Python files in the examples directory."""
    files = _example_files()
    assert files, f"No Python files found in {EXAMPLES_DIR}"


def test_examples_directory_has_ollama_files():
    """Fail if there are fewer than two Ollama example files in the examples directory."""
    ollama_files = [p for p in _example_files() if _is_ollama_example(p)]
    assert len(ollama_files) >= 2, (
        f"Expected at least two Ollama example files, found {len(ollama_files)}: "
        f"{[p.relative_to(EXAMPLES_DIR).as_posix() for p in ollama_files]}"
    )


def pytest_generate_tests(metafunc):
    """Generate test parameters dynamically from the examples directory."""
    if "example_path" in metafunc.fixturenames:
        example_files = _example_files()
        ids = [p.relative_to(EXAMPLES_DIR).as_posix() for p in example_files]
        metafunc.parametrize("example_path", example_files, ids=ids)


def test_example_syntax(example_path):
    """Assert that the example file has valid Python syntax, a module docstring,
    an ``if __name__ == '__main__':`` guard, and valid imports.

    The source is parsed with ``ast.parse`` and then compiled with ``compile``
    to ensure it is syntactically valid and can be transformed into a code
    object without executing it.
    """
    source = example_path.read_text(encoding="utf-8")
    # Parse only; do not execute to avoid any API calls or side effects.
    tree = ast.parse(source, filename=str(example_path))
    # Compile the AST to bytecode to catch any remaining compile-time issues.
    compile(tree, str(example_path), "exec")

    rel_path = example_path.relative_to(EXAMPLES_DIR).as_posix()

    assert _has_docstring(tree), f"{rel_path} is missing a module docstring."
    assert _has_main_guard(tree), f"{rel_path} is missing an if __name__ == '__main__' guard."
    assert _has_valid_imports(tree), f"{rel_path} contains invalid imports."


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
