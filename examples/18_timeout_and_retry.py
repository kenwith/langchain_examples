import random
import time
from typing import Callable, TypeVar, Any

T = TypeVar("T")


def exponential_backoff_with_jitter(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter_factor: float = 0.5,
) -> float:
    """Calculate sleep time with exponential backoff and jitter.

    Args:
        attempt: The current retry attempt (0-indexed).
        base_delay: Base delay in seconds.
        max_delay: Maximum delay in seconds.
        jitter_factor: Fraction of the delay to use as random jitter.

    Returns:
        Sleep time in seconds.
    """
    exponential_delay = min(max_delay, base_delay * (2 ** attempt))
    jitter = random.uniform(0, exponential_delay * jitter_factor)
    return exponential_delay + jitter


def retry_with_backoff(
    func: Callable[..., T],
    *args: Any,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter_factor: float = 0.5,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    **kwargs: Any,
) -> T:
    """Retry a function with exponential backoff and jitter.

    Args:
        func: The function to retry.
        *args: Positional arguments for func.
        max_retries: Maximum number of retries after the first attempt.
        base_delay: Base delay in seconds.
        max_delay: Maximum delay in seconds.
        jitter_factor: Fraction of the delay to use as random jitter.
        exceptions: Tuple of exception types to catch and retry.
        **kwargs: Keyword arguments for func.

    Returns:
        The result of func.

    Raises:
        The last exception if all retries are exhausted.
    """
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except exceptions as e:
            if attempt == max_retries:
                raise
            sleep_time = exponential_backoff_with_jitter(
                attempt, base_delay, max_delay, jitter_factor
            )
            print(f"Attempt {attempt + 1} failed: {e}. Retrying in {sleep_time:.2f}s...")
            time.sleep(sleep_time)
    raise RuntimeError("Unreachable")  # type: ignore[unreachable]


# Example usage
def flaky_operation() -> str:
    """Simulate an operation that fails often."""
    if random.random() < 0.7:
        raise ConnectionError("Simulated network error")
    return "Success"


if __name__ == "__main__":
    random.seed(42)
    print("Starting flaky operation with retry...")
    result = retry_with_backoff(flaky_operation, max_retries=5)
    print(f"Result: {result}")
