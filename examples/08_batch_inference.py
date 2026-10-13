"""Batch inference example using LangChain."""

import asyncio
from typing import Any, Dict, List, Union

from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate


async def run_batch(prompts: List[str], llm_chain: LLMChain) -> List[Dict[str, Union[str, bool]]]:
    """Process multiple inputs concurrently using asyncio.

    Args:
        prompts: A list of prompt strings to process.
        llm_chain: An LLMChain instance with async support.

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

    results = await asyncio.gather(*(process_prompt(p) for p in prompts))
    return list(results)


async def main() -> None:
    """Run a sample batch inference."""
    llm = OpenAI(temperature=0)
    prompt_template = PromptTemplate.from_template("Question: {question}\nAnswer:")
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)

    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in one sentence.",
    ]
    results = await run_batch(prompts, llm_chain)
    for res in results:
        print(res)


if __name__ == "__main__":
    asyncio.run(main())
