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
- Rate limit errors are detected and retried with exponential backoff.
- Async invocation with retry using asyncio to complement synchronous usage.

| Concept          | Implementation                                  |
|------------------|-------------------------------------------------|
| Model creation   | `init_chat_model(model="...")`                  |
| Timeout          | `model.with_timeout(30)`                        |
| Retry            | `model.with_retry(stop_after_attempt=3, ...)`   |
| Backoff          | `wait_exponential_jitter=True`                  |
| Rate limit handling | `is_rate_limit_error()` + retry on provider rate limit exceptions |
| Sync error handling | try/except around `model.invoke(...)`        |
| Async error handling | try/except around `await model.ainvoke(...)` |

Run the example with: `python examples/18_timeout_and_retry.py`
"""

import asyncio
import os
from langchain.chat_models import init_chat_model


def _get_rate_limit_exception_types():
    """
    Collect rate limit exception types from optional provider packages.

    This keeps the example provider-agnostic: if a provider library is installed,
    its rate limit error will be included in the retry tuple. If not, we simply
    skip it and still retry on timeouts and connection errors.

    Returns:
        tuple: Rate limit exception types (possibly empty).
    """
    rate_limit_exceptions = []

    try:
        from openai import RateLimitError
        rate_limit_exceptions.append(RateLimitError)
    except ImportError:
        pass

    try:
        from anthropic import RateLimitError
        rate_limit_exceptions.append(RateLimitError)
    except ImportError:
        pass

    return tuple(rate_limit_exceptions)


def is_rate_limit_error(e: Exception) -> bool:
    """
    Best-effort detection of rate limit errors across providers.

    Args:
        e: The exception raised by the model call.

    Returns:
        bool: True if the exception looks like a rate limit error.
    """
    if type(e).__name__ == "RateLimitError":
        return True
    if getattr(e, "status_code", None) == 429:
        return True
    if "rate limit" in str(e).lower():
        return True
    return False


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

    # Retry on common transient errors: timeouts, connection errors, and
    # provider-specific rate limit errors (if the provider package is installed).
    # Use exponential backoff with jitter: each retry waits longer than the
    # previous one, and the jitter spreads out retries across concurrent calls.
    retry_exceptions = (TimeoutError, ConnectionError) + _get_rate_limit_exception_types()
    model = model.with_retry(
        stop_after_attempt=max_attempts,
        retry_if_exception_type=retry_exceptions,
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
        if is_rate_limit_error(e):
            print(f"Rate limit error after retries: {e}\n")
        else:
            print(f"Error after retries: {e}\n")


async def ainvoke_with_retry(question: str, model) -> str:
    """
    Asynchronously ask a question using the model, relying on the configured retry policy.

    This is the async complement to `ask_question`; it uses `await model.ainvoke(...)`
    to leverage the same timeout and retry configuration.

    Args:
        question: The prompt to send to the model.
        model: The configured chat model (wrapped with timeout and retry).

    Returns:
        str: The model's response content.

    Raises:
        Exception: If the call fails after all retries.
    """
    response = await model.ainvoke(question)
    return response.content


async def async_main() -> None:
    """Demonstrate the same model with async invocation and retry."""
    model = get_model_with_retry()
    print(f"Async using model: {model}\n")

    questions = [
        "What is LangChain?",
        "Explain the benefits of retrying transient errors.",
    ]

    for q in questions:
        try:
            content = await ainvoke_with_retry(q, model)
            print(f"Q: {q}")
            print(f"A: {content}\n")
        except Exception as e:
            print(f"Q: {q}")
            if is_rate_limit_error(e):
                print(f"Rate limit error after async retries: {e}\n")
            else:
                print(f"Error after async retries: {e}\n")


def main() -> None:
    """Demonstrate a model with timeout and retry (synchronous)."""
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
    print("--- Running async example ---\n")
    asyncio.run(async_main())
