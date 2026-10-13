"""Tests for the Ollama async LangGraph agent supervisor example.

| Test | Description |
|------|-------------|
| test_ollama_async_langgraph_agent_supervisor | Validates the example runs end-to-end with Ollama. |
"""

import re
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

EXAMPLE_FILENAME = "ollama_async_langgraph_agent_supervisor.py"
EXAMPLE_PATH = Path(__file__).resolve().parent.parent / "examples" / EXAMPLE_FILENAME


def _get_example_model(example_path: Path) -> str | None:
    """Extract the model name from the example's init_chat_model call."""
    try:
        text = example_path.read_text()
    except OSError:
        return None
    match = re.search(
        r'init_chat_model\([^)]*model=["\']([^"\']+)["\']', text, re.DOTALL
    )
    return match.group(1) if match else None


def _ollama_available() -> bool:
    """Return True if Ollama is reachable and can run the example's model."""
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5) as response:
            if response.status != 200:
                return False
    except Exception:
        return False

    model = _get_example_model(EXAMPLE_PATH)
    if model is None:
        model = "llama3.1"

    try:
        from langchain.chat_models import init_chat_model

        chat_model = init_chat_model(model=model, provider="ollama")
        chat_model.invoke("ping")
        return True
    except Exception:
        return False


def test_ollama_async_langgraph_agent_supervisor() -> None:
    """Run the supervisor example and assert it completes successfully."""
    if not _ollama_available():
        pytest.skip("Ollama is not available or the example model is not installed")

    assert EXAMPLE_PATH.exists(), f"Example file not found: {EXAMPLE_PATH}"

    result = subprocess.run(
        [sys.executable, str(EXAMPLE_PATH)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"Example failed with stderr:\n{result.stderr}"


if __name__ == "__main__":
    test_ollama_async_langgraph_agent_supervisor()
