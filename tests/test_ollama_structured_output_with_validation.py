"""Tests for the Ollama structured output example.

This test verifies that the Ollama structured output pattern using
provider-agnostic ``init_chat_model`` returns a validated Pydantic object
with non-empty fields. It skips automatically if an Ollama server is not
reachable at ``localhost:11434``.
"""

import os
import socket
from pathlib import Path
import sys

import pytest
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

# Allow importing example modules from the repository root if needed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# | Test Function | Description |
# |---------------|-------------|
# | test_ollama_structured_output_returns_valid_data | Verifies that the Ollama structured output returns a validated Pydantic object. |


class Joke(BaseModel):
    """A joke with a setup and punchline."""

    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline of the joke")


def _ollama_available() -> bool:
    """Return True if an Ollama server is reachable on localhost:11434."""
    try:
        with socket.create_connection(("localhost", 11434), timeout=2):
            return True
    except OSError:
        return False


@pytest.mark.skipif(
    not _ollama_available(),
    reason="Ollama server is not available on localhost:11434",
)
def test_ollama_structured_output_returns_valid_data() -> None:
    """Test that Ollama structured output validates and returns structured data."""
    model_name = os.getenv("OLLAMA_MODEL", "llama3.1")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    llm = init_chat_model(
        model_name,
        provider="ollama",
        base_url=base_url,
    )
    structured_llm = llm.with_structured_output(Joke)

    result = structured_llm.invoke("Tell me a programming joke.")

    assert isinstance(result, Joke)
    assert result.setup.strip(), "Setup should not be empty"
    assert result.punchline.strip(), "Punchline should not be empty"


if __name__ == "__main__":
    # Small demo block to run the test directly with `python tests/test_ollama_structured_output_with_validation.py`
    if not _ollama_available():
        print("Ollama server not available; skipping test.")
        raise SystemExit(0)

    test_ollama_structured_output_returns_valid_data()
    print("Ollama structured output test passed.")
