"""Test for the Ollama async RAG example.

| Test Name                | Description                                         |
|--------------------------|-----------------------------------------------------|
| test_ollama_async_rag    | Verifies the async RAG example runs end-to-end      |
|                          | using a fake model and in-memory vector store.      |
"""

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure the repo root is on sys.path so "examples" is importable.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _fake_chat_model():
    """Return a fake async chat model that responds with a fixed answer."""
    model = MagicMock()
    model.ainvoke = AsyncMock(
        return_value=MagicMock(
            content="The answer is 42. This is a fake response from the RAG pipeline."
        )
    )
    return model


def _fake_embeddings():
    """Return fake embeddings for testing."""
    embeddings = MagicMock()
    embeddings.aembed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
    embeddings.aembed_documents = AsyncMock(
        return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    )
    return embeddings


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_ollama_async_rag_example(tmp_path):
    """Run the async RAG example with fake model/embeddings and verify output."""
    # Import here so missing dependencies only fail when the test is run,
    # and we can skip gracefully if the example is not present.
    try:
        from examples.ollama_async_rag import main
    except ImportError as exc:
        pytest.skip(f"Ollama async RAG example not available: {exc}")

    # Patch the example's init_chat_model and embeddings creation so that
    # no real Ollama server is required.
    with patch("examples.ollama_async_rag.init_chat_model", return_value=_fake_chat_model()), \
         patch("examples.ollama_async_rag.OllamaEmbeddings", return_value=_fake_embeddings()):

        # The example's main() is async and expects a query and a vector store
        # path. We point it to a temporary directory.
        result = await main(
            query="What is the meaning of life?",
            persist_directory=str(tmp_path / "vectorstore"),
        )

    assert result is not None
    assert "fake response" in result.lower()


# ---------------------------------------------------------------------------
# Demo / manual run
# ---------------------------------------------------------------------------
def _demo():
    """Run the test logic manually and print the result."""
    async def _run():
        test_dir = Path(tempfile.mkdtemp())
        try:
            await test_ollama_async_rag_example(test_dir)
        except pytest.skip.Exception as exc:
            print(f"Skipped: {exc}")
        except Exception as exc:
            print(f"Failed: {exc}")
        else:
            print("OK: Ollama async RAG example works with fake dependencies.")

    asyncio.run(_run())


if __name__ == "__main__":
    import tempfile

    _demo()
