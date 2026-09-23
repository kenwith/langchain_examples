"""
Batch inference with LangChain.

This example demonstrates how to run multiple prompts through an LLM in
batches, limiting the number of concurrent requests. It uses asyncio to
manage concurrency and includes retry logic with exponential backoff.

Batch size:
    Set the `batch_size` argument in `run_batch` to control how many
    prompts are processed concurrently. For example, `batch_size=5`
    processes at most 5 prompts at a time.

Output format:
    `run_batch` returns a list of strings with the same length as the
    input `prompts`. Each element is the model's response for the
    corresponding prompt. If a prompt fails after all retries, its
    result is the string `"Error: <message>"`.

    `process_batch` is a convenience wrapper around `run_batch` that
    prints each result with its original index in the input list.
"""

import asyncio
import random
from typing import Any, Callable, List, Optional

from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split a list into equal-sized chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


async def _run_with_retry(
    chain: LLMChain,
    prompt: str,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> str:
    """Run a single prompt with retries and exponential backoff."""
    for attempt in range(max_retries):
        try:
            return await chain.arun(prompt=prompt)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
            print(f"Error processing prompt, retrying in {delay:.2f}s: {e}")
            await asyncio.sleep(delay)
    raise RuntimeError("Unreachable retry state")  # pragma: no cover


async def run_batch(
    prompts: List[str],
    llm: Any = None,
    batch_size: int = 10,
    max_retries: int = 3,
    progress_callback: Optional[Callable[[int, int, Optional[str]], None]] = None,
) -> List[str]:
    """Run prompts through an LLM in batches, guarding against empty input.

    Args:
        prompts: List of prompt strings.
        llm: Language model instance. Defaults to OpenAI(temperature=0).
        batch_size: Maximum number of prompts to process concurrently.
        max_retries: Number of attempts per prompt before failing.
        progress_callback: Optional callback called as (completed, total, error)
            after each prompt completes. `error` is None on success, or the
            error message string on failure.

    Returns:
        List of strings, one per input prompt, in the same order. On
        failure after retries, the corresponding string is
        "Error: <error message>".
    """
    if not prompts:
        return []

    if llm is None:
        llm = OpenAI(temperature=0)

    prompt_template = PromptTemplate(
        input_variables=["prompt"],
        template="Answer the following question:\n{prompt}",
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)

    results: List[str] = [""] * len(prompts)
    completed = 0
    total = len(prompts)

    for batch_start in range(0, len(prompts), batch_size):
        batch = prompts[batch_start:batch_start + batch_size]
        batch_indices = list(range(batch_start, batch_start + len(batch)))

        tasks = {}
        for idx, prompt in zip(batch_indices, batch):
            task = asyncio.ensure_future(
                _run_with_retry(chain, prompt, max_retries)
            )
            tasks[task] = idx

        for completed_task in asyncio.as_completed(tasks):
            idx = tasks[completed_task]
            error = None
            try:
                result = await completed_task
                results[idx] = result
            except Exception as e:
                error = str(e)
                results[idx] = f"Error: {e}"
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, total, error)

    return results


async def process_batch(
    prompts: List[str],
    llm: Any = None,
    batch_size: int = 10,
    max_retries: int = 3,
    progress_callback: Optional[Callable[[int, int, Optional[str]], None]] = None,
) -> List[str]:
    """Run prompts through an LLM in batches and print each result with its index.

    This is a convenience wrapper around `run_batch` that prints each result
    alongside its original index in the input list. It returns early if the
    input list is empty.

    Args:
        prompts: List of prompt strings.
        llm: Language model instance. Defaults to OpenAI(temperature=0).
        batch_size: Maximum number of prompts to process concurrently.
        max_retries: Number of attempts per prompt before failing.
        progress_callback: Optional callback called as (completed, total, error)
            after each prompt completes.

    Returns:
        List of strings, one per input prompt, in the same order. If `prompts`
        is empty, returns an empty list.
    """
    if not prompts:
        return []

    results = await run_batch(
        prompts,
        llm=llm,
        batch_size=batch_size,
        max_retries=max_retries,
        progress_callback=progress_callback,
    )

    for idx, result in enumerate(results):
        print(f"[{idx}] {result}")

    return results


async def main():
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a haiku about Python.",
        "What are the benefits of using LangChain?",
        "Tell me a fun fact about space.",
    ]

    def show_progress(done: int, total: int, error: Optional[str] = None) -> None:
        if error:
            print(f"Progress: {done}/{total} (error: {error})")
        else:
            print(f"Progress: {done}/{total}")

    results = await run_batch(
        prompts,
        batch_size=2,
        progress_callback=show_progress,
    )

    for prompt, result in zip(prompts, results):
        print(f"Q: {prompt}\nA: {result}\n")


if __name__ == "__main__":
    asyncio.run(main())
