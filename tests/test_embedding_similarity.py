"""Tests for the embedding similarity example.

| Test | Description |
|------|-------------|
| test_cosine_similarity_ordering | Verifies that cosine similarity scores order vectors correctly. |
| test_threshold_filtering | Verifies that threshold filtering keeps only scores above the cutoff. |
| test_empty_inputs | Verifies that empty inputs are handled gracefully. |
| test_init_chat_model_importable | Verifies that init_chat_model is available for provider-agnostic usage. |
"""

import os

import numpy as np
import pytest
from langchain.chat_models import init_chat_model

from examples.embedding_similarity import cosine_similarity, filter_by_threshold


def test_cosine_similarity_ordering():
    """Similarity scores should be highest for the most related vector."""
    query = np.array([[1.0, 0.0]])
    docs = np.array([
        [1.0, 0.0],   # same direction -> 1.0
        [0.0, 1.0],   # orthogonal -> 0.0
        [-1.0, 0.0],  # opposite -> -1.0
    ])

    similarities = cosine_similarity(query, docs)
    scores = similarities[0]

    assert scores[0] > scores[1] > scores[2]
    assert np.isclose(scores[0], 1.0)
    assert np.isclose(scores[1], 0.0)
    assert np.isclose(scores[2], -1.0)


def test_threshold_filtering():
    """Only scores above the threshold should be returned."""
    scores = [0.9, 0.5, 0.3, 0.8]
    threshold = 0.6
    filtered = filter_by_threshold(scores, threshold)

    assert filtered == [0.9, 0.8]


def test_empty_inputs():
    """Empty vector lists should yield no scores."""
    query = np.array([[1.0, 0.0]])
    docs = np.empty((0, 2))

    similarities = cosine_similarity(query, docs)
    scores = similarities[0]

    assert scores.size == 0


def test_init_chat_model_importable():
    """init_chat_model should be importable and callable."""
    assert callable(init_chat_model)


if __name__ == "__main__":
    # Demonstrate provider-agnostic chat model initialization.
    provider = os.getenv("CHAT_MODEL_PROVIDER", "openai")
    model = os.getenv("CHAT_MODEL_NAME", "gpt-4o-mini")
    print(f"Initializing chat model with provider={provider}, model={model}")
    # Uncomment to actually initialize:
    # chat = init_chat_model(model, provider=provider)
    # print(chat)
