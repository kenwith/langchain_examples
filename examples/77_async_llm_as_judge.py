"""
Example: Async LLM as Judge
===========================

| Field       | Value                                                                 |
|-------------|-----------------------------------------------------------------------|
| Description | Scores generated answers against a rubric using an LLM judge asynchronously. |
| Model       | Provider-agnostic via `init_chat_model`                               |
| Async       | Yes                                                                    |

This example demonstrates how to use a language model as a judge to evaluate
generated answers against a predefined rubric. The judge returns a numeric
score and feedback. The example is fully asynchronous and works with any
provider supported by `init_chat_model` (e.g., OpenAI, Anthropic, Groq, etc.).

The `judge_async` helper provides a single asynchronous entry point that
creates the judge model and scores an answer in one call. The lower-level
`create_judge` and `score_answer` functions are also available for custom
workflows.

To run this example, set the appropriate API key environment variable for the
chosen model provider (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GROQ_API_KEY`).
The default model is `"gpt-4o"` but can be overridden via the `MODEL_NAME`
environment variable.
"""

import asyncio
import json
import os
from typing import Any, Dict, Tuple

from langchain.chat_models import init_chat_model


def create_judge(model_name: str | None = None, temperature: float = 0.0) -> Any:
    """
    Create a chat model instance to act as the judge.

    Args:
        model_name: Optional model name. Defaults to the `MODEL_NAME`
            environment variable or `"gpt-4o"`.
        temperature: Sampling temperature. Defaults to 0.0 for deterministic scoring.

    Returns:
        A configured chat model instance.
    """
    if model_name is None:
        model_name = os.getenv("MODEL_NAME", "gpt-4o")
    return init_chat_model(model_name=model_name, temperature=temperature)


async def score_answer(
    question: str,
    answer: str,
    rubric: str,
    judge: Any,
) -> Tuple[int, str]:
    """
    Score an answer against a rubric using the LLM judge.

    Args:
        question: The question that was asked.
        answer: The generated answer to be scored.
        rubric: The scoring rubric.
        judge: A chat model instance used as the judge.

    Returns:
        A tuple containing the numeric score (0-10) and feedback text.
    """
    prompt = f"""
You are a strict and fair judge. Evaluate the provided answer against the given rubric.
Return a JSON object with two fields:
- "score": an integer from 0 to 10
- "feedback": a short explanation of the score

Question:
{question}

Answer:
{answer}

Rubric:
{rubric}

Your JSON response:
"""
    response = await judge.ainvoke(prompt)
    content = response.content

    # Attempt to parse JSON from the model output.
    try:
        # Try direct parsing first.
        result = json.loads(content)
    except json.JSONDecodeError:
        # Fallback: extract JSON from markdown code blocks if present.
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            result = json.loads(content)
        else:
            raise ValueError(f"Could not parse judge output as JSON: {content}")

    score = int(result.get("score", 0))
    feedback = result.get("feedback", "")
    return score, feedback


async def judge_async(
    question: str,
    answer: str,
    rubric: str,
    model_name: str | None = None,
    temperature: float = 0.0,
) -> Tuple[int, str]:
    """
    Asynchronously create a judge model and score an answer against a rubric.

    This is a convenience wrapper around `create_judge` and `score_answer`.

    Args:
        question: The question that was asked.
        answer: The generated answer to be scored.
        rubric: The scoring rubric.
        model_name: Optional model name. Defaults to the `MODEL_NAME`
            environment variable or `"gpt-4o"`.
        temperature: Sampling temperature. Defaults to 0.0 for deterministic scoring.

    Returns:
        A tuple containing the numeric score (0-10) and feedback text.
    """
    judge = create_judge(model_name, temperature)
    return await score_answer(question, answer, rubric, judge)


async def main() -> None:
    """
    Main async demo: score a sample answer using the `judge_async` helper.
    """
    # Example question, answer, and rubric.
    question = "What is the capital of France?"
    answer = "The capital of France is Paris."
    rubric = """
- Accuracy (5 points): The answer must be factually correct.
- Completeness (3 points): The answer should fully address the question.
- Clarity (2 points): The answer should be clear and concise.
"""

    print("Scoring answer...")
    score, feedback = await judge_async(question, answer, rubric)
    print(f"Score: {score}/10")
    print(f"Feedback: {feedback}")


if __name__ == "__main__":
    asyncio.run(main())
