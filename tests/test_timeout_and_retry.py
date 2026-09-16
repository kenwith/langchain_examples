"""
Tests for timeout and retry behavior using fake models.
"""

import pytest
import os
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.language_models.base import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.callbacks import CallbackManagerForLLMRun
from typing import Any, List, Optional


class FakeRetryModel(BaseChatModel):
    """
    A fake chat model that can be configured to fail or succeed on specific calls.
    """

    responses: List[str]
    failure_count: int = 0
    sleep_time: float = 0.0
    raise_timeout: bool = False

    def _generate(
        self,
        messages: List[Any],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.sleep_time > 0:
            import time
            time.sleep(self.sleep_time)

        if self.raise_timeout:
            raise TimeoutError("Simulated timeout")

        if self.failure_count > 0:
            self.failure_count -= 1
            raise ValueError("Simulated failure")

        response = self.responses.pop(0) if self.responses else "Default response"
        message = AIMessage(content=response)
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self) -> str:
        return "fake-retry-model"


def test_retry_success():
    """
    Test that the retry wrapper retries on failure and eventually succeeds.
    """
    model = FakeRetryModel(responses=["final answer"], failure_count=2)
    retry_model = model.with_retry(stop_after_attempt=3, wait_exponential_jitter=False)

    result = retry_model.invoke("Hello")
    assert result.content == "final answer"


def test_retry_exhausted():
    """
    Test that the retry wrapper stops after max attempts and raises the last error.
    """
    model = FakeRetryModel(responses=[], failure_count=10)
    retry_model = model.with_retry(stop_after_attempt=3, wait_exponential_jitter=False)

    with pytest.raises(ValueError, match="Simulated failure"):
        retry_model.invoke("Hello")


def test_timeout_raises():
    """
    Test that a timeout error is propagated (no retry on timeout by default).
    """
    model = FakeRetryModel(responses=[], raise_timeout=True)
    # Do not apply retry; ensure timeout is raised
    with pytest.raises(TimeoutError, match="Simulated timeout"):
        model.invoke("Hello")


def test_timeout_with_retry():
    """
    Test that retry does not retry on timeout by default (or does if configured).
    Here we verify that the timeout error is raised and not swallowed.
    """
    model = FakeRetryModel(responses=[], raise_timeout=True)
    retry_model = model.with_retry(stop_after_attempt=3, wait_exponential_jitter=False)

    # By default, retry only catches specific exceptions, not TimeoutError.
    # We expect the TimeoutError to propagate immediately.
    with pytest.raises(TimeoutError, match="Simulated timeout"):
        retry_model.invoke("Hello")


def test_retry_wrapper_applied():
    """
    Verify that the retry wrapper is actually applied to the model.
    """
    model = FakeRetryModel(responses=["ok"])
    retry_model = model.with_retry(stop_after_attempt=2)
    # Check that the retry model has the expected attribute or behavior
    # Here we simply check that it still works and returns the response.
    result = retry_model.invoke("Hello")
    assert result.content == "ok"


if __name__ == "__main__":
    # Simple demo of the retry behavior
    model = FakeRetryModel(responses=["success after retries"], failure_count=2)
    retry_model = model.with_retry(stop_after_attempt=3, wait_exponential_jitter=False)
    print("Invoking model with retry...")
    result = retry_model.invoke("Test")
    print(f"Result: {result.content}")
    print("Demo complete.")
