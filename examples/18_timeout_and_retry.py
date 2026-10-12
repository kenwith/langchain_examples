import functools
import random
import time
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def exponential_backoff_with_jitter(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter_factor: float = 1.0,
) -> float:
    """Calculate sleep time with exponential backoff and full jitter.

    Full jitter randomizes the sleep time between 0 and the exponential
    backoff value, which helps prevent thundering herd problems when many
    clients retry simultaneously.

    Args:
        attempt: The current retry attempt (0-indexed).
        base_delay: Base delay in seconds.
        max_delay: Maximum delay in seconds.
        jitter_factor: Fraction of the exponential delay to use as the
            upper bound for random jitter. Defaults to 1.0 for full jitter.

    Returns:
        Sleep time in seconds.
    """
    exponential_delay = min(max_delay, base_delay * (2 ** attempt))
    max_sleep = min(exponential_delay * jitter_factor, max_delay)
    return random.uniform(0, max_sleep)


def retry_with_backoff(
    func: Callable[..., T],
    *args: Any,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter_factor: float = 1.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    should_retry: Callable[[Exception], bool] | None = None,
    **kwargs: Any,
) -> T:
    """Retry a function with exponential backoff and jitter.

    Args:
        func: The function to retry.
        *args: Positional arguments for func.
        max_retries: Maximum number of retries after the first attempt.
        base_delay: Base delay in seconds.
        max_delay: Maximum delay in seconds.
        jitter_factor: Fraction of the exponential delay to use as the
            upper bound for random jitter. Defaults to 1.0 for full jitter.
        exceptions: Tuple of exception types to catch and retry. Ignored if
            should_retry is provided.
        should_retry: Optional callable that takes an exception and returns
            True if the failure is retryable, False otherwise. When provided,
            this overrides the exceptions parameter.
        **kwargs: Keyword arguments for func.

    Returns:
        The result of func.

    Raises:
        The last exception if all retries are exhausted or a non-retryable
        failure occurs.
    """

    def is_retryable(exc: Exception) -> bool:
        if should_retry is not None:
            return should_retry(exc)
        return isinstance(exc, exceptions)

    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries or not is_retryable(e):
                raise
            sleep_time = exponential_backoff_with_jitter(
                attempt, base_delay, max_delay, jitter_factor
            )
            print(f"Attempt {attempt + 1} failed: {e}. Retrying in {sleep_time:.2f}s...")
            time.sleep(sleep_time)
    raise RuntimeError("Unreachable")  # type: ignore[unreachable]


def retryable(
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter_factor: float = 1.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    should_retry: Callable[[Exception], bool] | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Return a decorator that retries a function with exponential backoff.

    Args:
        max_retries: Maximum number of retries after the first attempt.
        base_delay: Base delay in seconds.
        max_delay: Maximum delay in seconds.
        jitter_factor: Fraction of the exponential delay to use as the
            upper bound for random jitter. Defaults to 1.0 for full jitter.
        exceptions: Tuple of exception types to catch and retry. Ignored if
            should_retry is provided.
        should_retry: Optional callable that takes an exception and returns
            True if the failure is retryable, False otherwise.

    Returns:
        A decorator that wraps the given function with retry logic.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return retry_with_backoff(
                func,
                *args,
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
                jitter_factor=jitter_factor,
                exceptions=exceptions,
                should_retry=should_retry,
                **kwargs,
            )

        return wrapper

    return decorator


# Example usage
def flaky_operation() -> str:
    """Simulate an operation that fails often."""
    if random.random() < 0.7:
        raise ConnectionError("Simulated network error")
    return "Success"


@retryable(max_retries=5, exceptions=(ConnectionError,))
def decorated_flaky_operation() -> str:
    """Simulate an operation that fails often, with retry applied."""
    if random.random() < 0.7:
        raise ConnectionError("Simulated network error")
    return "Success"


if __name__ == "__main__":
    random.seed(42)

    print("Starting flaky operation with retry (direct call)...")
    result = retry_with_backoff(
        flaky_operation,
        max_retries=5,
        exceptions=(ConnectionError,),
    )
    print(f"Result: {result}")

    print("Starting flaky operation with retry (decorator)...")
    result = decorated_flaky_operation()
    print(f"Result: {result}")
