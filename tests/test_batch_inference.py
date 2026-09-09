"""Unit tests for the batch inference helper.

| Test | Description | Expected Result |
|------|-------------|-----------------|
| test_batch_inference_returns_expected_count | Verifies that the batch helper returns one result per input prompt. | The number of results equals the number of prompts. |
| test_batch_inference_empty_input | Verifies that the batch helper gracefully handles an empty prompt list. | An empty list is returned. |
"""

from types import SimpleNamespace
from unittest.mock import patch

from batch_inference import batch_inference
from langchain.chat_models import init_chat_model


class FakeChatModel:
    """A minimal fake chat model for testing without API calls."""

    def __init__(self, responses):
        self.responses = responses
        self.invoke_calls = []
        self.batch_calls = []

    def invoke(self, prompt):
        self.invoke_calls.append(prompt)
        index = len(self.invoke_calls) - 1
        return SimpleNamespace(content=self.responses[index])

    def batch(self, prompts):
        self.batch_calls.append(prompts)
        return [SimpleNamespace(content=response) for response in self.responses]


def test_batch_inference_returns_expected_count():
    """Verify that the batch helper returns one result per input prompt."""
    fake_model = FakeChatModel(["response1", "response2", "response3"])
    prompts = ["prompt1", "prompt2", "prompt3"]

    with patch("langchain.chat_models.init_chat_model", return_value=fake_model):
        model = init_chat_model("fake-model")
        results = batch_inference(model, prompts)

    assert len(results) == len(prompts)
    assert results == ["response1", "response2", "response3"]


def test_batch_inference_empty_input():
    """Verify that the batch helper handles an empty prompt list."""
    fake_model = FakeChatModel([])

    with patch("langchain.chat_models.init_chat_model", return_value=fake_model):
        model = init_chat_model("fake-model")
        results = batch_inference(model, [])

    assert results == []


if __name__ == "__main__":
    test_batch_inference_returns_expected_count()
    test_batch_inference_empty_input()
    print("All batch inference tests passed!")
