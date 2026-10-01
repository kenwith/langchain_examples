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
models that expose an `invoke` method.

Example usage:
    from my_model import model
    prompts = ["Hello", "World"]
    results = batch_infer(model.invoke, prompts, max_concurrency=5)

    # Or with batch_predict:
    results = batch_predict(model, prompts, max_concurrency=5)

Run the demo with:
    python 08_batch_inference.py
"""

import asyncio
from typing import Any, Callable, List, TypeVar

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
