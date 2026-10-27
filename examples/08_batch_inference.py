"""Batch inference example using LangChain."""

import asyncio
from typing import Callable, Dict, List, Optional, Union

from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate


async def run_batch(
    prompts: List[str],
    llm_chain: LLMChain,
    batch_size: int = 2,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[Dict[str, Union[str, bool]]]:
    """Process multiple inputs in batches using asyncio.

    Args:
        prompts: A list of prompt strings to process.
        llm_chain: An LLMChain instance with async support.
        batch_size: Maximum number of prompts to process concurrently per batch.
            If zero or negative, all prompts are processed in a single batch.
        progress_callback: An optional callback invoked after each batch with
            the number of completed prompts and the total number of prompts.

    Returns:
        A list of dictionaries, each containing the prompt, result or error,
        and a success flag.
    """
    async def process_prompt(prompt: str) -> Dict[str, Union[str, bool]]:
        try:
            result = await llm_chain.arun(prompt)
            return {"prompt": prompt, "result": result, "success": True}
        except Exception as e:
            return {"prompt": prompt, "error": str(e), "success": False}

    total = len(prompts)
    if total == 0:
        return []

    effective_batch_size = batch_size if batch_size > 0 else total
    results = []
    for start in range(0, total, effective_batch_size):
        batch = prompts[start:start + effective_batch_size]
        batch_results = await asyncio.gather(*(process_prompt(p) for p in batch))
        results.extend(batch_results)
        if progress_callback:
            progress_callback(len(results), total)

    return results


def format_batch_results(results: List[Dict[str, Union[str, bool]]]) -> str:
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
        prompts, llm_chain, batch_size=1, progress_callback=show_progress
    )
    print(format_batch_results(results))


if __name__ == "__main__":
    asyncio.run(main())
