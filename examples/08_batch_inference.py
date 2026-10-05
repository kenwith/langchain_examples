#!/usr/bin/env python3
"""
Module for batch inference.

This module provides reusable helper functions for running batch inference
concurrently using asyncio. It is particularly useful for improving throughput
when calling LLM APIs, as multiple prompts can be processed in parallel without
blocking the event loop.

The main public function is `batch_infer`, which takes a synchronous inference
function and a list of inputs, and returns the outputs in the same order as the
inputs. A convenience wrapper `batch_predict` is also provided for LangChain-style
models that expose an `invoke` method. For models that support native batching,
`batch_generate` sends all prompts in a single model call. For large lists of
inputs, `process_batch` is a generator that yields results with progress logging.

Example usage:
    from my_model import model
    prompts = ["Hello", "World"]
    results = batch_infer(model.invoke, prompts, max_concurrency=5)

    # Or with batch_predict:
    results = batch_predict(model, prompts, max_concurrency=5)

    # Or with batch_generate for one native batched call:
    responses = batch_generate(model, prompts)

    # Or with process_batch for progress logging:
    for result in process_batch(model.invoke, prompts, batch_size=5):
        print(result)

Run the demo with:
    python 08_batch_inference.py
"""

import asyncio
import logging
from typing import Any, Callable, Iterator, List, TypeVar

# Type variable for input and output types
T = TypeVar("T")
U = TypeVar("U")


async def _async_infer(
    infer_func: Callable[[T], U], input_item: T, semaphore: asyncio.Semaphore
) -> U:
    """Run a single inference, respecting a semaphore limit."""
    async with semaphore:
        # Run the synchronous inference function in a thread to avoid blocking
        # the event loop.
        return await asyncio.to_thread(infer_func, input_item)


async def _batch_infer_async(
    infer_func: Callable[[T], U],
    inputs: List[T],
    max_concurrency: int = 10,
) -> List[U]:
    """Asynchronous implementation of batch inference.

    Args:
        infer_func: A synchronous function that takes a single input and returns
            an output.
        inputs: A list of inputs to process.
        max_concurrency: Maximum number of concurrent inference calls.

    Returns:
        A list of outputs in the same order as inputs.
    """
    semaphore = asyncio.Semaphore(max_concurrency)

    async def bounded_infer(item: T) -> U:
        return await _async_infer(infer_func, item, semaphore)

    # asyncio.gather preserves the order of the inputs.
    return await asyncio.gather(*(bounded_infer(item) for item in inputs))


def batch_infer(
    infer_func: Callable[[T], U],
    inputs: List[T],
    max_concurrency: int = 10,
) -> List[U]:
    """Process multiple inputs concurrently using asyncio.

    This is a blocking function that runs an asyncio event loop to execute
    `infer_func` on each input concurrently. The `infer_func` should be a
    synchronous function (e.g., a LangChain model's `invoke` method). If your
    inference function is already async, you can pass it directly, but this
    wrapper is designed for synchronous callables.

    Args:
        infer_func: A synchronous function that maps a single input to an output.
        inputs: A list of inputs to be processed.
        max_concurrency: Maximum number of simultaneous inference calls. Useful
            for rate limiting or resource management.

    Returns:
        A list of outputs, preserving the order of `inputs`.

    Raises:
        Exception: If any inference call fails, the exception is propagated.

    Example:
        >>> from my_langchain_model import model
        >>> results = batch_infer(model.invoke, ["Hello", "World"], max_concurrency=5)
    """
    return asyncio.run(_batch_infer_async(infer_func, inputs, max_concurrency))


def batch_predict(
    model: Any,
    prompts: List[str],
    max_concurrency: int = 10,
) -> List[Any]:
    """Run batch predictions using a LangChain-style model.

    This is a convenience wrapper around `batch_infer` for models that expose
    an `invoke` method. It allows you to pass the model object directly instead
    of binding the method manually.

    Args:
        model: An object with an `invoke` method that takes a single prompt
            and returns a prediction.
        prompts: A list of prompt strings.
        max_concurrency: Maximum number of concurrent predictions.

    Returns:
        A list of predictions in the same order as `prompts`.

    Example:
        >>> from my_langchain_model import model
        >>> results = batch_predict(model, ["Hello", "World"], max_concurrency=5)
    """
    return batch_infer(model.invoke, prompts, max_concurrency)


def batch_generate(
    model: Any,
    prompts: List[str],
) -> List[str]:
    """Generate responses for multiple prompts in a single model call.

    This helper uses a LangChain model's `generate` method, which sends all
    prompts to the underlying API at once. This can be more efficient than
    making multiple `invoke` calls, especially for models that support native
    batching.

    Args:
        model: An object with a `generate` method that accepts a list of
            prompts and returns an LLMResult.
        prompts: A list of prompt strings.

    Returns:
        A list of generated response strings, in the same order as `prompts`.

    Example:
        >>> from my_langchain_model import model
        >>> responses = batch_generate(model, ["Hello", "World"])
    """
    result = model.generate(prompts)
    return [gen[0].text for gen in result.generations]


def process_batch(
    infer_func: Callable[[T], U],
    inputs: List[T],
    batch_size: int = 10,
    max_concurrency: int = 10,
    log_interval: int = 1,
) -> Iterator[U]:
    """Process inputs in batches, yielding results with progress logging.

    This generator processes a large list of inputs in smaller batches, calling
    `batch_infer` on each batch. Progress is logged every `log_interval` batches
    using the `logging` module, which is useful for long-running jobs.

    Args:
        infer_func: A synchronous function that takes a single input and returns
            an output.
        inputs: A list of inputs to process.
        batch_size: Number of inputs to process per batch. Defaults to 10.
        max_concurrency: Maximum number of concurrent inference calls per batch.
            Passed to `batch_infer`. Defaults to 10.
        log_interval: Log progress every N batches. Defaults to 1.

    Yields:
        The output for each input, in the same order as `inputs`.

    Raises:
        ValueError: If `batch_size` or `log_interval` is not positive.

    Example:
        >>> results = list(process_batch(model.invoke, prompts, batch_size=5))
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if log_interval <= 0:
        raise ValueError("log_interval must be positive")

    total = len(inputs)
    if total == 0:
        return

    batch_count = 0
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = inputs[start:end]
        batch_results = batch_infer(infer_func, batch, max_concurrency=max_concurrency)
        for result in batch_results:
            yield result

        batch_count += 1
        processed = end
        if batch_count % log_interval == 0 or processed == total:
            logging.info("Processed %d/%d inputs", processed, total)


# --- Example usage ---------------------------------------------------------
def example_infer(text: str) -> str:
    """Mock inference function for demonstration purposes.

    In a real scenario, this would call an LLM or model. Here we just simulate
    asynchronous work by sleeping a bit.
    """
    import random
    import time

    delay = random.uniform(0.1, 0.5)
    time.sleep(delay)  # Simulate synchronous I/O-bound work
    return f"Processed: {text} (slept {delay:.2f}s)"


def main():
    """Run a simple demonstration of batch_infer."""
    inputs = [
        "What is the capital of France?",
        "Explain quantum computing.",
        "Write a poem about a tree.",
        "What's the weather today?",
        "Give me three good books.",
        "How to make pasta?",
        "Who is Nikola Tesla?",
        "Define artificial intelligence.",
    ]

    # Use a smaller concurrency to illustrate the semaphore effect
    results = batch_infer(example_infer, inputs, max_concurrency=3)

    # Print results
    for original, processed in zip(inputs, results):
        print(f"Input: {original}\nOutput: {processed}\n")


if __name__ == "__main__":
    main()
