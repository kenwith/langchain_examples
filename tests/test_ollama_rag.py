"""Tests for the Ollama RAG example.

This test verifies that examples/41_ollama_rag.py runs correctly without a
local Ollama server by mocking `init_chat_model` and `OllamaEmbeddings`.

| Test | Description |
|------|-------------|
| test_main_with_mocked_ollama | Runs main() with mocked LLM and embeddings |
"""

from pathlib import Path
import importlib.util
from unittest.mock import MagicMock

import pytest


def load_example_module():
    """Load the 41_ollama_rag.py example as a module."""
    example_path = Path(__file__).resolve().parent.parent / "examples" / "41_ollama_rag.py"
    spec = importlib.util.spec_from_file_location("ollama_rag_example", example_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_main_with_mocked_ollama(monkeypatch):
    """Run the example main() with mocked Ollama dependencies."""
    example = load_example_module()

    # Mock chat model (provider-agnostic init_chat_model)
    mock_chat_model = MagicMock()
    mock_chat_model.invoke.return_value = MagicMock(content="Mocked response")

    # Mock OllamaEmbeddings
    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents.return_value = [[0.0, 0.1, 0.2], [0.1, 0.2, 0.3]]
    mock_embeddings.embed_query.return_value = [0.0, 0.1, 0.2]

    monkeypatch.setattr(example, "init_chat_model", lambda *args, **kwargs: mock_chat_model)
    monkeypatch.setattr(example, "OllamaEmbeddings", lambda *args, **kwargs: mock_embeddings)

    # main() should run without raising or requiring a local Ollama server
    example.main()


if __name__ == "__main__":
    pytest.main([__file__])
