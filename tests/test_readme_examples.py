import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"

# Files that are part of the project infrastructure, not examples.
NON_EXAMPLE_FILES = {
    "setup.py",
    "conftest.py",
    "__init__.py",
}

# Directories that should never be scanned for examples.
NON_EXAMPLE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "tests",
}


def is_example_file(path: Path) -> bool:
    """Return True if the path looks like an example script, not infrastructure."""
    if path.name in NON_EXAMPLE_FILES:
        return False
    if path.name.startswith("test_") or path.name.endswith("_test.py"):
        return False
    if any(part in NON_EXAMPLE_DIRS for part in path.parts):
        return False
    return path.suffix == ".py"


def get_example_files():
    """Return all example .py files in the repository."""
    return {path.relative_to(ROOT) for path in ROOT.rglob("*.py") if is_example_file(path)}


def get_readme_references():
    """Return all .py filenames/paths mentioned in the README."""
    text = README.read_text(encoding="utf-8")
    # Match any string that looks like a Python file name, possibly with directories.
    refs = set(re.findall(r"[\w./\-]+\.py", text))
    normalized = set()
    for ref in refs:
        ref = ref.replace("\\", "/")
        if ref.startswith("./"):
            ref = ref[2:]
        normalized.add(ref)
    return normalized


def test_readme_lists_existing_examples():
    """Every .py file referenced in the README must exist in the repository."""
    example_files = get_example_files()
    example_basenames = {p.name for p in example_files}
    example_paths = {p.as_posix() for p in example_files}

    for ref in get_readme_references():
        if "/" in ref:
            # If the README gives a path, that exact path must exist.
            assert ref in example_paths, f"README references {ref}, but it does not exist."
        else:
            # If the README gives a bare filename, at least one file with that name must exist.
            assert ref in example_basenames, f"README references {ref}, but no such example file exists."


def test_all_examples_are_listed_in_readme():
    """Every example file in the repository must be mentioned in the README."""
    readme_references = get_readme_references()
    readme_basenames = {Path(ref).name for ref in readme_references}
    readme_paths = {ref for ref in readme_references if "/" in ref}

    missing = []
    for path in get_example_files():
        posix_path = path.as_posix()
        if posix_path in readme_paths or path.name in readme_basenames:
            continue
        missing.append(posix_path)

    assert not missing, f"Example files missing from README: {missing}"
