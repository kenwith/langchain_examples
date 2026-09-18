"""
Example 18: Timeout and Retry
============================

This example demonstrates how to configure timeouts and retries for chat models
in a provider-agnostic way using `init_chat_model`, `with_timeout`, and `with_retry`.

Key Concepts
------------
- Provider-agnostic model initialization via `init_chat_model`.
- Enforce hard timeouts on model calls with `with_timeout`.
- Automatically retry on transient errors using `with_retry`.
- Use exponential backoff and jitter to avoid overwhelming the provider during retries.
- Consistent error handling and graceful degradation.

| Concept          | Implementation                                  |
|------------------|-------------------------------------------------|
| Model creation   | `init_chat_model(model="...")`                  |
| Timeout          | `model.with_timeout(30)`                        |
| Retry            | `model.with_retry(stop_after_attempt=3, ...)`   |
| Backoff          | `wait_exponential_jitter=True`                  |
| Error handling   | try/except around `model.invoke(...)`           |

Run the example with: `python examples/18_timeout_and_retry.py`
"""

import os
from langchain.chat_models import init_chat_model


def get_model_with_retry(timeout: int = 30, max_attempts: int = 3):
    """
    Create a provider-agnostic chat model with timeout and exponential backoff retry.

    Args:
        timeout: Maximum time in seconds for a single model call.
        max_attempts: Number of retry attempts after the initial call.

    Returns:
        Runnable: A chat model wrapped with timeout and retry.
    """
    # By default, init_chat_model infers the provider from the model name.
    # It will use environment variables (e.g., OPENAI_API_KEY, ANTHROPIC_API_KEY)
    # to authenticate automatically. Do not hardcode keys.
    model = init_chat_model(
        model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
        temperature=0,
    )

    # Enforce a hard timeout on every invocation.
    model = model.with_timeout(timeout)

    # Retry on common transient errors: timeouts and connection errors.
    # Use exponential backoff with jitter: each retry waits longer than the
    # previous one, and the jitter spreads out retries across concurrent calls.
    model = model.with_retry(
        stop_after_attempt=max_attempts,
        retry_if_exception_type=(TimeoutError, ConnectionError),
        wait_exponential_jitter=True,
    )

    return model


def ask_question(question: str, model) -> None:
    """
    Ask a question and handle any errors that may still occur.

    Args:
        question: The prompt to send to the model.
        model: The configured chat model.
    """
    try:
        response = model.invoke(question)
        print(f"Q: {question}")
        print(f"A: {response.content}\n")
    except Exception as e:
        print(f"Q: {question}")
        print(f"Error after retries: {e}\n")


def main() -> None:
    """Demonstrate a model with timeout and retry."""
    model = get_model_with_retry()
    print(f"Using model: {model}\n")

    questions = [
        "What is LangChain?",
        "Explain the benefits of retrying transient errors.",
    ]

    for q in questions:
        ask_question(q, model)


if __name__ == "__main__":
    main()
