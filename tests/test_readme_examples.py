"""Test that every example file in the examples directory is mentioned in the README."""

from pathlib import Path
import pytest

# Repo root is two levels up from this test file
REPO_ROOT = Path(__file__).resolve().parents[2]
README_PATH = REPO_ROOT / "README.md"
EXAMPLES_DIR = REPO_ROOT / "examples"

def test_all_example_files_are_mentioned_in_readme():
    """Fail if any .py file under examples/ is not referenced in README.md."""
    readme_text = README_PATH.read_text(encoding="utf-8")

    # Collect all Python files in examples (recursively)
    example_files = sorted(EXAMPLES_DIR.rglob("*.py"))
    assert example_files, f"No example files found under {EXAMPLES_DIR}"

    missing = []
    for example_file in example_files:
        # Use relative path as the canonical name that should appear in README
        rel_path = example_file.relative_to(REPO_ROOT)
        # Also check just the filename in case README uses only basename
        if str(rel_path) not in readme_text and example_file.name not in readme_text:
            missing.append(rel_path)

    assert not missing, (
        f"Example file(s) not mentioned in README: {missing}\n"
        "Please add a reference to each example file in README.md."
    )

if __name__ == "__main__":
    # Simple manual runner for this test
    import sys
    test_all_example_files_are_mentioned_in_readme()
    print("All example files are mentioned in README.md")
    sys.exit(0)
