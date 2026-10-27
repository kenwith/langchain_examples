"""Example 18: Timeout and retry with exponential backoff and jitter."""

import functools
import logging
import random
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class RetryableError(Exception):
    """Custom exception for retryable operations."""


def retry_with_exponential_backoff(
    max_retries=5,
    base_delay=0.5,
    max_delay=10.0,
    jitter=0.5,
    timeout=30.0,
    exceptions=(RetryableError,),
):
    """Retry a function with exponential backoff, jitter, and an overall timeout.

    Args:
        max_retries: Maximum number of retries before giving up.
        base_delay: Initial delay before the first retry (seconds).
        max_delay: Maximum delay between retries (seconds).
        jitter: Maximum random additional delay (seconds).
        timeout: Overall timeout for all attempts (seconds).
        exceptions: Tuple of exception classes that should trigger a retry.

    Returns:
        Decorated function.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.monotonic()
            attempt = 0

            while True:
                # Check overall timeout before each attempt
                elapsed = time.monotonic() - start_time
                if elapsed >= timeout:
                    raise TimeoutError(
                        f"Operation timed out after {timeout:.2f} seconds"
                    )

                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    attempt += 1
                    if attempt > max_retries:
                        logger.error("Giving up after %d attempts", attempt - 1)
                        raise

                    # Exponential backoff with full jitter
                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    delay = random.uniform(0, delay + jitter)

                    logger.warning(
                        "Attempt %d failed with %s. Retrying in %.2f seconds...",
                        attempt,
                        exc,
                        delay,
                    )
                    time.sleep(delay)

        return wrapper

    return decorator


def operation_that_may_fail(attempt_number):
    """Simulate an operation that fails intermittently."""
    if attempt_number < 3:
        raise RetryableError("Temporary failure")
    return "Success"


@retry_with_exponential_backoff(
    max_retries=5,
    base_delay=0.1,
    max_delay=1.0,
    jitter=0.2,
    timeout=10.0,
)
def flaky_operation():
    """A flaky operation that succeeds after a few attempts."""
    attempt = getattr(flaky_operation, "attempt", 0) + 1
    flaky_operation.attempt = attempt
    logger.info("Executing flaky_operation, attempt %d", attempt)
    return operation_that_may_fail(attempt)


def main():
    """Run the example."""
    try:
        result = flaky_operation()
        logger.info("Final result: %s", result)
    except Exception as exc:
        logger.error("Operation failed: %s", exc)


if __name__ == "__main__":
    main()
