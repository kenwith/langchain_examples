"""
LangChain evaluation example: criteria-based evaluation plus a custom evaluator
that checks for keyword presence and length constraints.

This module demonstrates two evaluation criteria:
1. LLM-based relevance scoring via LangChain's criteria evaluator.
2. Deterministic keyword and length checks via a custom StringEvaluator.
"""

import os
from typing import List, Optional

from langchain.evaluation import load_evaluator
from langchain.evaluation.schema import StringEvaluator
from langchain_openai import ChatOpenAI


class KeywordAndLengthEvaluator(StringEvaluator):
    """
    Custom evaluator that applies two criteria:
    - Keyword coverage: all required keywords must appear (case-insensitive).
    - Length sufficiency: prediction must exceed a minimum character count.

    Returns a binary score (1 if both criteria pass, otherwise 0) and an explanation.
    """

    def __init__(self, keywords: List[str], min_length: int):
        self.keywords = [kw.lower() for kw in keywords]
        self.min_length = min_length

    @property
    def requires_input(self) -> bool:
        # This evaluator does not use the input prompt.
        return False

    @property
    def requires_reference(self) -> bool:
        # This evaluator does not use a reference answer.
        return False

    @property
    def evaluation_name(self) -> str:
        return "keyword_and_length"

    def _evaluate_strings(
        self, prediction: str, reference: str = None, input: str = None, **kwargs
    ) -> tuple[float, str]:
        # Length check
        length_ok = len(prediction) > self.min_length

        # Keyword check
        lower_prediction = prediction.lower()
        missing_keywords = [kw for kw in self.keywords if kw not in lower_prediction]
        all_keywords_present = len(missing_keywords) == 0

        score = 1 if (length_ok and all_keywords_present) else 0

        reasons = []
        if not length_ok:
            reasons.append(f"prediction too short (min {self.min_length} chars)")
        if missing_keywords:
            reasons.append(f"missing keyword(s): {missing_keywords}")

        explanation = (
            "Passed." if score == 1 else "Failed. " + "; ".join(reasons)
        )
        return score, explanation


def evaluate_response(
    prediction: str,
    reference: str,
    input_prompt: str,
    llm: ChatOpenAI,
    criteria: str = "relevance",
    keywords: Optional[List[str]] = None,
    min_length: Optional[int] = None,
) -> dict:
    """
    Evaluate a response using two complementary criteria:

    1. Built-in LLM-based criteria evaluation (e.g., relevance).
    2. Custom keyword and length evaluation, if both keywords and min_length are provided.

    Args:
        prediction: The generated response to evaluate.
        reference: An optional reference answer for comparison.
        input_prompt: The original prompt used to generate the response.
        llm: The LLM instance used by the criteria evaluator.
        criteria: The name of the built-in criteria to use (default: "relevance").
        keywords: List of required keywords for the custom evaluator.
        min_length: Minimum character length for the custom evaluator.

    Returns:
        A dictionary with keys "criteria_result" and "custom_result".
        "custom_result" is None if keywords or min_length is not provided.
    """
    results = {}

    criteria_evaluator = load_evaluator(
        "criteria",
        criteria=criteria,
        llm=llm,
    )
    results["criteria_result"] = criteria_evaluator.evaluate_strings(
        prediction=prediction,
        reference=reference,
        input=input_prompt,
    )

    if keywords is not None and min_length is not None:
        custom_evaluator = KeywordAndLengthEvaluator(
            keywords=keywords,
            min_length=min_length,
        )
        results["custom_result"] = custom_evaluator.evaluate_strings(
            prediction=prediction,
            reference=reference,
            input=input_prompt,
        )
    else:
        results["custom_result"] = None

    return results


def main():
    # Initialise the LLM (used by the criteria evaluator).
    # The API key is read from the environment only – never hard-code it here.
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Example prediction and reference for demonstration
    prediction = "I enjoy programming in Python and learning about clean code."
    reference = "A good answer mentions Python and programming."
    input_prompt = "What do you think about programming?"

    results = evaluate_response(
        prediction=prediction,
        reference=reference,
        input_prompt=input_prompt,
        llm=llm,
        criteria="relevance",
        keywords=["python", "programming"],
        min_length=50,
    )

    # 1) Built‑in criteria evaluation (e.g., relevance)
    print("=== Criteria (relevance) result ===")
    for key, value in results["criteria_result"].items():
        print(f"{key}: {value}")

    # 2) Custom evaluator: keyword presence + length constraint
    print("\n=== Custom (keyword + length) result ===")
    for key, value in results["custom_result"].items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
