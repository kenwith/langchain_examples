"""Tests for the Ollama reranking example.

This module verifies that the reranking example works with mocked embeddings
and falls back gracefully when the cross-encoder is missing.

| Test | Description |
|------|-------------|
| test_reranking_with_mocked_embeddings | Verifies reranking works with mocked embeddings. |
| test_reranking_falls_back_without_cross_encoder | Verifies graceful fallback when cross-encoder is unavailable. |
"""

from unittest.mock import patch

from langchain_examples.ollama_reranking import rerank_documents


@patch("langchain_examples.ollama_reranking.CrossEncoder")
@patch("langchain_examples.ollama_reranking.OllamaEmbeddings")
def test_reranking_with_mocked_embeddings(mock_embeddings, mock_cross_encoder):
    """Verify reranking works when embeddings and cross-encoder are mocked."""
    mock_embeddings.return_value.embed_query.return_value = [0.1, 0.2]
    mock_embeddings.return_value.embed_documents.return_value = [
        [0.1, 0.2],
        [0.2, 0.1],
    ]
    mock_cross_encoder.return_value.predict.return_value = [0.9, 0.1]

    docs = ["doc1", "doc2"]
    result = rerank_documents("query", docs)

    assert result == ["doc1", "doc2"]


@patch("langchain_examples.ollama_reranking.CrossEncoder", side_effect=ImportError)
@patch("langchain_examples.ollama_reranking.OllamaEmbeddings")
def test_reranking_falls_back_without_cross_encoder(mock_embeddings, mock_cross_encoder):
    """Verify reranking falls back to original order when cross-encoder is missing."""
    mock_embeddings.return_value.embed_query.return_value = [0.1, 0.2]
    mock_embeddings.return_value.embed_documents.return_value = [
        [0.1, 0.2],
        [0.2, 0.1],
    ]

    docs = ["doc1", "doc2"]
    result = rerank_documents("query", docs)

    assert result == docs


if __name__ == "__main__":
    from langchain.chat_models import init_chat_model

    try:
        model = init_chat_model("ollama/llama2", temperature=0)
        print(f"Initialized chat model: {model.__class__.__name__}")
    except Exception as exc:
        print(f"Could not initialize chat model: {exc}")
