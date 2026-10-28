"""Tests for syntax of example files."""
from pathlib import Path
import ast
import re
import pytest

# Directory containing example files
EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"

# Patterns for obvious hardcoded credentials
CREDENTIAL_PATTERNS = [
    re.compile(r"(?i)(password|passwd|pwd)\s*[=:]\s*['\"][^'\"]+['\"]"),
    re.compile(r"(?i)(api[_-]?key|apikey|secret|token)\s*[=:]\s*['\"][^'\"]+['\"]"),
    re.compile(r"(?i)(aws_access_key_id|aws_secret_access_key)\s*[=:]\s*['\"][^'\"]+['\"]"),
]

def test_example_syntax():
    """Check that all example files are syntactically valid and have docstrings."""
    example_files = sorted(EXAMPLES_DIR.glob("*.py"))
    assert example_files, f"No example files found in {EXAMPLES_DIR}"
    for file_path in example_files:
        # Read source
        source = file_path.read_text(encoding="utf-8")
        # Parse and compile to catch syntax errors
        try:
            tree = ast.parse(source)
            compile(source, str(file_path), "exec")
        except SyntaxError as e:
            pytest.fail(f"Syntax error in {file_path}: {e}")
        # Check module docstring
        if not ast.get_docstring(tree):
            pytest.fail(f"Missing module docstring in {file_path}")
        # Check for hardcoded credentials
        for pattern in CREDENTIAL_PATTERNS:
            if pattern.search(source):
                pytest.fail(f"Hardcoded credential pattern found in {file_path}: {pattern.pattern}")
