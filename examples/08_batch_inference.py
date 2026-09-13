import asyncio
from typing import List, Any

from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split a list into equal-sized chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


async def run_batch(prompts: List[str], llm: Any = None, batch_size: int = 10) -> List[str]:
    """Run prompts through an LLM in batches, guarding against empty input."""
    if not prompts:
        return []

    if llm is None:
        llm = OpenAI(temperature=0)

    prompt_template = PromptTemplate(
        input_variables=["prompt"],
        template="Answer the following question:\n{prompt}",
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)

    results: List[str] = []
    for batch in chunk_list(prompts, batch_size):
        try:
            batch_results = await asyncio.gather(
                *(chain.arun(prompt=p) for p in batch)
            )
        except Exception as e:
            print(f"Error processing batch: {e}")
            raise
        results.extend(batch_results)

    return results


async def main():
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a haiku about Python.",
        "What are the benefits of using LangChain?",
        "Tell me a fun fact about space.",
    ]

    results = await run_batch(prompts, batch_size=2)

    for prompt, result in zip(prompts, results):
        print(f"Q: {prompt}\nA: {result}\n")


if __name__ == "__main__":
    asyncio.run(main())
