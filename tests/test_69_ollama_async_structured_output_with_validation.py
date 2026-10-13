"""Test for the Ollama async structured output with validation example.

This test verifies that the example's main async function returns a structured
output that conforms to the expected Pydantic schema. The test is skipped if
the required Ollama environment variables are not set or if the example module
cannot be imported.

## Test Overview
| Test Name | Description |
|-----------|-------------|
| test_ollama_async_structured_output_with_validation | Runs the example's main function and asserts the result matches the expected schema. |
"""

import os
import sys
import pytest

# Ensure the repository root is on the path so the example module can be imported.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ollama_async_structured_output_with_validation import main, Joke
except ImportError:
    # Fallback to the examples package if the module is not at the root.
    try:
        from examples.ollama_async_structured_output_with_validation import main, Joke
    except ImportError:
        main = None
        Joke = None


@pytest.mark.asyncio
async def test_ollama_async_structured_output_with_validation():
    """Run the example's main async function and validate the structured output."""
    if main is None or Joke is None:
        pytest.skip("Example module not found; skipping test.")

    # Ensure Ollama is configured. The example uses OLLAMA_BASE_URL to connect.
    if not os.getenv("OLLAMA_BASE_URL"):
        pytest.skip("OLLAMA_BASE_URL environment variable not set; skipping test.")

    # Execute the example's main function.
    result = await main()

    # Assert that the returned object matches the expected schema.
    assert isinstance(result, Joke), (
        f"Expected result to be an instance of {Joke.__name__}, "
        f"but got {type(result).__name__}."
    )


if __name__ == "__main__":
    pytest.main([__file__])
