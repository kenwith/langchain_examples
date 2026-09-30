"""Tests for the Ollama embeddings example.

| Test Name | Description |
|-----------|-------------|
| test_similarity_search | Verifies similarity search runs with a fake embedding class. |
"""

from ollama_embeddings import similarity_search


class FakeEmbeddings:
    """A fake embedding class that returns deterministic vectors."""

    def embed_documents(self, texts):
        """Return a list of embeddings for the given texts."""
        return [[1.0, 0.0] if "apple" in text else [0.0, 1.0] for text in texts]

    def embed_query(self, text):
        """Return an embedding for the given query."""
        return [1.0, 0.0] if "fruit" in text else [0.0, 1.0]


def test_similarity_search():
    """Test that similarity search returns the expected document."""
    documents = ["apple", "banana"]
    query = "fruit"
    result = similarity_search(FakeEmbeddings(), query, documents)
    assert result == "apple"


if __name__ == "__main__":
    test_similarity_search()
    print("Test passed.")
