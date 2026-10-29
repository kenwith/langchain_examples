"""Example 86: Ollama async LLM as judge.

This example demonstrates an asynchronous evaluation flow using two local
Ollama models. One model generates answers to questions, while a second
model acts as a judge and provides a verdict on each answer.

Uses provider-agnostic ``init_chat_model`` from LangChain to instantiate
both models. The generator and judge models can be selected via environment
variables, defaulting to ``llama3`` for both.
"""

import asyncio
import os

from langchain.chat_models import init_chat_model

GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "llama3")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "llama3")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


async def generate_answer(question: str) -> str:
    """Generate an answer using the local Ollama generator model."""
    generator = init_chat_model(
        model=GENERATOR_MODEL,
        model_provider="ollama",
        base_url=OLLAMA_BASE_URL,
        temperature=0.2,
    )
    response = await generator.ainvoke([("human", question)])
    return str(response.content)


async def judge_answer(question: str, answer: str) -> str:
    """Judge the quality of an answer using the local Ollama judge model."""
    judge = init_chat_model(
        model=JUDGE_MODEL,
        model_provider="ollama",
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )
    prompt = (
        "You are a judge. Evaluate the following answer for the question. "
        "Reply with a score from 1 to 10 and a brief explanation.\n\n"
        f"Question: {question}\n\nAnswer: {answer}"
    )
    response = await judge.ainvoke([("human", prompt)])
    return str(response.content)


async def run_evaluation(questions: list[str]) -> None:
    """Run the generate-and-judge flow for each question."""
    for question in questions:
        answer = await generate_answer(question)
        verdict = await judge_answer(question, answer)
        print(f"Question: {question}\n")
        print(f"Answer: {answer}\n")
        print(f"Verdict: {verdict}\n")
        print("=" * 50)


if __name__ == "__main__":
    sample_questions = [
        "What is LangChain?",
        "Explain asynchronous programming in Python.",
    ]
    asyncio.run(run_evaluation(sample_questions))
