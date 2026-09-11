"""
LangChain evaluation example: criteria-based evaluation plus a custom evaluator
that checks for keyword presence and length constraints.
"""

import os
from typing import List

from langchain.evaluation import load_evaluator
from langchain.evaluation.schema import StringEvaluator
from langchain_openai import ChatOpenAI


class KeywordAndLengthEvaluator(StringEvaluator):
    """
    Custom evaluator that checks:
    - Whether required keywords appear (case‑insensitive) in the prediction.
    - Whether the prediction meets a minimum character length.
    Returns a binary score (1/0) and an explanation.
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

    # 1) Built‑in criteria evaluation (e.g., relevance)
    criteria_evaluator = load_evaluator(
        "criteria",
        criteria="relevance",
        llm=llm,
    )
    criteria_result = criteria_evaluator.evaluate_strings(
        prediction=prediction,
        reference=reference,
        input=input_prompt,
    )
    print("=== Criteria (relevance) result ===")
    for key, value in criteria_result.items():
        print(f"{key}: {value}")

    # 2) Custom evaluator: keyword presence + length constraint
    custom_evaluator = KeywordAndLengthEvaluator(
        keywords=["python", "programming"],
        min_length=50,
    )
    custom_result = custom_evaluator.evaluate_strings(
        prediction=prediction,
        reference=reference,
        input=input_prompt,
    )
    print("\n=== Custom (keyword + length) result ===")
    for key, value in custom_result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
