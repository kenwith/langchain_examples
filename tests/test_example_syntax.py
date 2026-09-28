"""Tests for example syntax.

This test module parses every Python file in the examples directory using
the ``ast`` module to catch syntax errors without executing any API calls.
It deliberately does not import or invoke ``init_chat_model`` to avoid
triggering network requests or requiring provider credentials.

In addition to valid syntax, every example must contain a module docstring
and an ``if __name__ == '__main__':`` guard so it can be safely imported
and run as a script.

| Test | Description |
|------|-------------|
| test_example_syntax | Parse every example file with ``ast`` and verify it has a module docstring and a main guard. |
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
    """Assert that the example file has valid Python syntax, a module docstring,
    and an ``if __name__ == '__main__':`` guard.
    """
    source = example_path.read_text(encoding="utf-8")
    # Parse only; do not execute to avoid any API calls or side effects.
    tree = ast.parse(source, filename=str(example_path))

    rel_path = example_path.relative_to(EXAMPLES_DIR).as_posix()

    assert _has_docstring(tree), f"{rel_path} is missing a module docstring."
    assert _has_main_guard(tree), f"{rel_path} is missing an if __name__ == '__main__' guard."


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
