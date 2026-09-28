#!/usr/bin/env python3
"""
Module for batch inference.

This module provides a utility function `batch_infer` that processes multiple
inputs concurrently using asyncio, which is useful for improving throughput
when calling LLM APIs. The module demonstrates how to use asyncio with
LangChain models to process a list of prompts efficiently.

Example usage:
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
    tasks = [
        asyncio.ensure_future(_async_infer(infer_func, item, semaphore))
        for item in inputs
    ]
    results: List[U] = []

    # Use as_completed to process results as they finish, but ensure ordering
    # by mapping each completed task to its original position.
    pending = {task: idx for idx, task in enumerate(tasks)}
    for completed_task in asyncio.as_completed(tasks):
        # Find the original index of this task
        idx = None
        for task, i in list(pending.items()):
            if task is completed_task:
                idx = i
                del pending[task]
                break
        if idx is None:
            raise RuntimeError("Task not found in pending mapping")

        result = await completed_task
        # Place result in the correct position (we'll build a list with None
        # placeholders for now)
        results.append((idx, result))

    # Sort results by index to restore order
    results.sort(key=lambda x: x[0])
    return [res for _, res in results]


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
