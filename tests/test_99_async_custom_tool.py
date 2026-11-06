"""Tests for the async custom tool example."""

import os
import sys
from pathlib import Path

import pytest

# Add the examples directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))


def test_async_custom_tool_example():
    """Run the async custom tool example and verify it completes without errors."""
    # Skip if no API key is set for any provider
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        pytest.skip("No API key available for testing the example")

    # Import the example module
    try:
        from async_custom_tool import main
    except ImportError as exc:
        pytest.fail(f"Could not import async_custom_tool: {exc}")

    # Run the main function
    main()


if __name__ == "__main__":
    pytest.main([__file__])
