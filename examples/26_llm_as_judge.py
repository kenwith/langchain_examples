"""
Example 26: LLM as Judge
Use a chat model to score a generated answer against a simple rubric.

This example demonstrates how to leverage a chat model to evaluate the quality
of a generated answer based on a predefined rubric. The model returns a score
and a brief justification, which is then printed to the console.

The example is provider-agnostic: it uses `init_chat_model` to instantiate a
model based on environment variables or defaults. Ensure your API keys are set
in the environment (e.g., OPENAI_API_KEY for OpenAI models).
"""

import os
import re
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------------
# Configuration (optional overrides via environment variables)
# ---------------------------------------------------------------------------
MODEL_NAME = os.getenv("LANGCHAIN_MODEL", "gpt-4o")
MODEL_PROVIDER = os.getenv("LANGCHAIN_MODEL_PROVIDER", "openai")

DEFAULT_RUBRIC = (
    "Evaluate the answer on a scale from 0 to 5, where 5 is the best.\n"
    "Consider the following criteria:\n"
    "- Correctness: Is the answer factually accurate?\n"
    "- Completeness: Does the answer fully address the question?\n"
    "- Clarity: Is the answer easy to understand and well-structured?\n"
    "A score of 5 indicates the answer is correct, complete, and clear. "
    "A score of 0 indicates the answer is entirely incorrect or irrelevant."
)


def build_evaluation_chain(
    model_name: str = MODEL_NAME,
    model_provider: str = MODEL_PROVIDER,
):
    """
    Build a LangChain runnable that evaluates an answer against a rubric.

    The chain expects a dictionary with keys:
        - "question": the original question
        - "answer": the generated answer
        - "rubric": the scoring criteria

    Args:
        model_name: Name of the chat model to use.
        model_provider: Provider of the chat model (e.g., "openai", "anthropic").

    Returns:
        A Runnable that accepts a dictionary and returns the judgment string.
    """
    model = init_chat_model(model_name, model_provider=model_provider)

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(
                    "You are an impartial judge. Evaluate the given answer strictly "
                    "according to the provided rubric. Return your response in the "
                    "following format:\n"
                    "Score: <number>/5\nJustification: <brief explanation>\n"
                    "The score must be a number from 0 to 5, where 5 is the best."
                )
            ),
            HumanMessage(
                content=(
                    "Question: {question}\n\n"
                    "Generated Answer: {answer}\n\n"
                    "Rubric: {rubric}"
                )
            ),
        ]
    )

    return prompt | model


def judge_answer(
    question: str,
    generated_answer: str,
    rubric: str = DEFAULT_RUBRIC,
    model_name: str = MODEL_NAME,
    model_provider: str = MODEL_PROVIDER,
) -> str:
    """
    Score a generated answer against a rubric using a chat model.

    Args:
        question: The original question.
        generated_answer: The answer to be judged.
        rubric: A description of the scoring criteria. Defaults to DEFAULT_RUBRIC.
        model_name: Name of the chat model to use.
        model_provider: Provider of the chat model (e.g., "openai", "anthropic").

    Returns:
        A string containing the model's judgment (score and justification).
    """
    chain = build_evaluation_chain(model_name, model_provider)
    response = chain.invoke(
        {
            "question": question,
            "answer": generated_answer,
            "rubric": rubric,
        }
    )
    return response.content


def parse_score(judgment: str, max_score: int = 5) -> float:
    """
    Extract the numeric score from a judgment string like "Score: 4/5".

    Args:
        judgment: The raw judgment output from the model.
        max_score: The expected maximum score (default: 5).

    Returns:
        The numeric score as a float.

    Raises:
        ValueError: If the score cannot be parsed or the denominator does not match.
    """
    pattern = r"Score:\s*(\d+(?:\.\d+)?)\s*/\s*(\d+)"
    match = re.search(pattern, judgment)
    if not match:
        raise ValueError(f"Could not parse score from judgment: {judgment!r}")
    score = float(match.group(1))
    total = int(match.group(2))
    if total != max_score:
        raise ValueError(
            f"Parsed denominator {total} does not match expected max_score {max_score}"
        )
    return score


def main():
    """Run a simple demo of the LLM-as-judge functionality."""
    question = "What is the capital of France?"
    generated_answer = "Paris is the capital of France."

    print("=== LLM as Judge Demo ===\n")
    print(f"Question: {question}")
    print(f"Generated Answer: {generated_answer}\n")

    judgment = judge_answer(question, generated_answer)
    print("Judgment:")
    print(judgment)

    # Demonstrate the parse_score helper
    try:
        score = parse_score(judgment)
        print(f"\nParsed Score: {score}/5")
    except ValueError as e:
        print(f"\nCould not parse score: {e}")


if __name__ == "__main__":
    main()
