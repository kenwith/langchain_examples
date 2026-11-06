"""Batch inference example using LangChain.

This module shows how to run batch inference with LangChain in a
provider-agnostic way. The core function, `run_batch`, relies only on the
asynchronous interface of `LLMChain`, so the same batch logic can be used
with any LangChain-compatible LLM, whether from OpenAI, Anthropic, Hugging
Face, or another provider.

The module also includes formatting helpers to present batch results in a
uniform way, covering both success and error cases, and a display function
for a clean, human-readable output.
"""

import asyncio
from typing import Callable, Dict, List, Optional, Union

from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate


async def run_batch(
    prompts: List[str],
    llm_chain: LLMChain,
    batch_size: int = 2,
    max_concurrency: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[Dict[str, Union[str, bool]]]:
    """Process multiple inputs in batches using asyncio.

    Args:
        prompts: A list of prompt strings to process.
        llm_chain: An LLMChain instance with async support.
        batch_size: Maximum number of prompts to process concurrently per batch.
            Must be a positive integer.
        max_concurrency: Maximum number of prompts processed concurrently across
            all batches. If None or non-positive, no limit is applied.
        progress_callback: An optional callback invoked after each batch with
            the number of completed prompts and the total number of prompts.
            If not provided, a default progress counter is used.

    Returns:
        A list of dictionaries, each containing the prompt, result or error,
        and a success flag.

    Raises:
        ValueError: If `prompts` is None or `batch_size` is not a positive integer.
    """
    if prompts is None:
        raise ValueError("prompts cannot be None")

    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    total = len(prompts)
    if total == 0:
        return []

    effective_batch_size = batch_size

    # Create a semaphore to limit overall concurrency
    if max_concurrency is not None and max_concurrency > 0:
        semaphore = asyncio.Semaphore(max_concurrency)
    else:
        semaphore = None

    # Use a default progress counter if none is provided
    if progress_callback is None:
        progress_callback = show_progress

    async def process_prompt(prompt: str) -> Dict[str, Union[str, bool]]:
        try:
            if semaphore is not None:
                async with semaphore:
                    result = await llm_chain.arun(prompt)
            else:
                result = await llm_chain.arun(prompt)
            return {"prompt": prompt, "result": result, "success": True}
        except Exception as e:
            return {"prompt": prompt, "error": str(e), "success": False}

    results = []
    for start in range(0, total, effective_batch_size):
        batch = prompts[start:start + effective_batch_size]
        batch_results = await asyncio.gather(*(process_prompt(p) for p in batch))
        results.extend(batch_results)
        progress_callback(len(results), total)

    return results


def _format_batch_results(results: List[Dict[str, Union[str, bool]]]) -> str:
    """Format batch inference results into a consistent string.

    Args:
        results: A list of result dictionaries from run_batch.

    Returns:
        A formatted string with one numbered entry per result.
    """
    lines = []
    for idx, res in enumerate(results, start=1):
        if res.get("success"):
            lines.append(f"{idx}. Prompt: {res['prompt']}\n   Result: {res['result']}")
        else:
            lines.append(f"{idx}. Prompt: {res['prompt']}\n   Error: {res['error']}")
    return "\n".join(lines)


def format_batch_results(results: List[Dict[str, Union[str, bool]]]) -> str:
    """Format batch inference results into a consistent string.

    This is a public wrapper around the internal `_format_batch_results`
    helper.

    Args:
        results: A list of result dictionaries from run_batch.

    Returns:
        A formatted string with one numbered entry per result.
    """
    return _format_batch_results(results)


def display_batch_results(results: List[Dict[str, Union[str, bool]]]) -> None:
    """Display batch inference results in a clean, human-readable format.

    This function prints a header, each result (or error), and a summary
    of successful and failed prompts.

    Args:
        results: A list of result dictionaries from run_batch.
    """
    print("Batch Inference Results")
    print("=======================")
    print(format_batch_results(results))
    successes = sum(1 for r in results if r.get("success"))
    failures = len(results) - successes
    print(f"\nSummary: {successes} succeeded, {failures} failed out of {len(results)} total.")


def show_progress(completed: int, total: int) -> None:
    """Print batch processing progress.

    Args:
        completed: The number of prompts completed so far.
        total: The total number of prompts to process.
    """
    print(f"Progress: {completed}/{total}")


async def main() -> None:
    """Run a sample batch inference."""
    llm = OpenAI(temperature=0)
    prompt_template = PromptTemplate.from_template("Question: {question}\nAnswer:")
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)

    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in one sentence.",
    ]
    results = await run_batch(
        prompts,
        llm_chain,
        batch_size=1,
        max_concurrency=2,
        # progress_callback is optional; if omitted, show_progress is used
    )
    display_batch_results(results)


if __name__ == "__main__":
    asyncio.run(main())
