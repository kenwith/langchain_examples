"""
Example: Fallback Chains for Model Errors.

This example demonstrates how to use the `run_with_fallback` helper to
handle errors from a model by trying a sequence of fallback callables.
If the primary model call raises an exception (e.g., rate limit, timeout),
the helper catches the error and invokes the next callable in the chain.
This pattern is useful for building robust LLM applications that can
gracefully degrade when a model provider fails.

Fallback model selection:
- Choose fallbacks that are independent from the primary provider, such as
  a different vendor, a local model, a cached response, or a heuristic rule.
- Order fallbacks from most desirable to least desirable. The helper stops
  at the first successful call, so the first fallback should be the one that
  best balances quality and availability.
- The `create_fallback_model` helper can generate simple fallback callables
  with a custom name and response prefix, which is convenient for testing
  and for scenarios where a lightweight response is acceptable.
- If all fallbacks fail, the last exception is re-raised.
"""

import time
from typing import Any, Callable, Sequence


class RateLimitError(Exception):
    """Raised when the model API rate limit is exceeded."""


def call_model(prompt: str) -> str:
    """Simulate a primary model call that sometimes fails."""
    if "test" in prompt:
        raise RateLimitError("Rate limit exceeded")
    return f"Model response to: {prompt}"


def call_fallback(prompt: str) -> str:
    """A simple heuristic fallback that always succeeds."""
    return f"Fallback response to: {prompt}"


def create_fallback_model(model_name: str, response_prefix: str) -> Callable[[str], str]:
    """
    Create a fallback model callable.

    Args:
        model_name: A name for the fallback model, used for identification.
        response_prefix: A prefix to include in the generated response.

    Returns:
        A callable that accepts a prompt and returns a simulated response.
    """
    def fallback_model(prompt: str) -> str:
        return f"{response_prefix} response to: {prompt}"

    fallback_model.__name__ = model_name
    return fallback_model


def run_with_fallback(
    funcs: Sequence[Callable[..., Any]], *args: Any, **kwargs: Any
) -> Any:
    """
    Try each callable in `funcs` in order.

    Args:
        funcs: A sequence of callables to try.
        *args, **kwargs: Arguments passed to each callable.

    Returns:
        The result of the first callable that does not raise an exception.

    Raises:
        The last exception if all callables fail.
    """
    last_exc: Exception | None = None
    for func in funcs:
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            print(f"Error: {exc}; trying next fallback...")
    if last_exc:
        raise last_exc
    raise RuntimeError("No callables provided")


def main() -> None:
    """Run the example fallback chain."""
    prompt = "test query"
    # Simulate rate limiting: wait and retry with fallback.
    wait_time = 0.5
    print("Rate limit exceeded, waiting %.2f seconds" % wait_time)
    time.sleep(wait_time)

    backup_model = create_fallback_model("backup", "Backup")
    result = run_with_fallback([call_model, backup_model, call_fallback], prompt)
    print(result)


if __name__ == "__main__":
    main()
