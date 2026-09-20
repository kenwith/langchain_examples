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
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

# ---------------------------------------------------------------------------
# Configuration (optional overrides via environment variables)
# ---------------------------------------------------------------------------
MODEL_NAME = os.getenv("LANGCHAIN_MODEL", "gpt-4o")
MODEL_PROVIDER = os.getenv("LANGCHAIN_MODEL_PROVIDER", "openai")


def judge_answer(
    question: str,
    generated_answer: str,
    rubric: str,
    model_name: str = MODEL_NAME,
    model_provider: str = MODEL_PROVIDER,
) -> str:
    """
    Score a generated answer against a rubric using a chat model.

    Args:
        question: The original question.
        generated_answer: The answer to be judged.
        rubric: A description of the scoring criteria.
        model_name: Name of the chat model to use.
        model_provider: Provider of the chat model (e.g., "openai", "anthropic").

    Returns:
        A string containing the model's judgment (score and justification).
    """
    # Instantiate the model
    model = init_chat_model(model_name, model_provider=model_provider)

    # System message sets the evaluation context
    system_message = SystemMessage(
        content=(
            "You are an impartial judge. Evaluate the given answer based on the "
            "provided rubric. Return your response in the following format:\n"
            "Score: <number>/5\nJustification: <brief explanation>"
        )
    )

    # User message contains the question, answer, and rubric
    user_message = HumanMessage(
        content=(
            f"Question: {question}\n\n"
            f"Generated Answer: {generated_answer}\n\n"
            f"Rubric: {rubric}"
        )
    )

    # Generate the judgment
    response = model.invoke([system_message, user_message])
    return response.content


def main():
    """Run a simple demo of the LLM-as-judge functionality."""
    question = "What is the capital of France?"
    generated_answer = "Paris is the capital of France."
    rubric = (
        "Score from 1 to 5 based on correctness, completeness, and clarity. "
        "A score of 5 means the answer is correct, complete, and clear."
    )

    print("=== LLM as Judge Demo ===\n")
    print(f"Question: {question}")
    print(f"Generated Answer: {generated_answer}\n")

    judgment = judge_answer(question, generated_answer, rubric)
    print("Judgment:")
    print(judgment)


if __name__ == "__main__":
    main()
