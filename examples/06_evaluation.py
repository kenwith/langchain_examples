"""
LangChain evaluation example: criteria-based evaluation plus custom evaluators
for deterministic checks, reference-answer scoring, and LLM-as-judge assessment.

This module demonstrates:
1. LLM-based relevance scoring via LangChain's criteria evaluator.
2. A direct LLM-as-judge evaluation with a custom prompt for quality scoring.
3. Deterministic keyword and length checks via a custom StringEvaluator.
4. Reference-answer token overlap scoring via a custom StringEvaluator.
5. A simple accuracy scorer for classification-style tasks.
6. A basic exact-match scorer for generated-response correctness.
"""

import os
import re
from typing import List, Optional

from langchain.evaluation import load_evaluator
from langchain.evaluation.schema import StringEvaluator
from langchain_openai import ChatOpenAI


def accuracy_scorer(predictions: List[str], references: List[str]) -> dict:
    """
    Compute simple accuracy and error count for classification tasks.

    Args:
        predictions: List of predicted labels/strings.
        references: List of ground truth labels/strings.

    Returns:
        A dictionary with keys:
            - "accuracy": fraction of exact matches (0.0 to 1.0)
            - "error_count": number of mismatches
            - "total": total number of samples

    Raises:
        ValueError: If predictions and references have different lengths.
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")

    total = len(predictions)
    if total == 0:
        return {"accuracy": 0.0, "error_count": 0, "total": 0}

    error_count = sum(1 for p, r in zip(predictions, references) if p != r)
    accuracy = (total - error_count) / total

    return {
        "accuracy": accuracy,
        "error_count": error_count,
        "total": total,
    }


def exact_match_scorer(predictions: List[str], references: List[str]) -> dict:
    """
    Compute the exact match rate for a set of prediction/reference pairs.

    This is a basic string-equality check and does not apply any
    normalisation. It is useful for tasks where the output must match a
    canonical answer exactly.

    Args:
        predictions: List of predicted strings.
        references: List of ground truth strings.

    Returns:
        A dictionary with keys:
            - "exact_match_rate": fraction of exact matches (0.0 to 1.0)
            - "exact_matches": number of exact matches
            - "total": total number of samples

    Raises:
        ValueError: If predictions and references have different lengths.

    Example:
        >>> exact_match_scorer(["python", "java"], ["python", "java"])
        {'exact_match_rate': 1.0, 'exact_matches': 2, 'total': 2}

        >>> exact_match_scorer(["python"], ["Java"])
        {'exact_match_rate': 0.0, 'exact_matches': 0, 'total': 1}
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")

    total = len(predictions)
    if total == 0:
        return {"exact_match_rate": 0.0, "exact_matches": 0, "total": 0}

    exact_matches = sum(1 for p, r in zip(predictions, references) if p == r)

    return {
        "exact_match_rate": exact_matches / total,
        "exact_matches": exact_matches,
        "total": total,
    }


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


class ReferenceAnswerEvaluator(StringEvaluator):
    """
    Custom evaluator that scores a prediction against a reference answer
    using token-level F1 overlap. The score is between 0 and 1.
    """

    @property
    def requires_input(self) -> bool:
        # This evaluator does not use the input prompt.
        return False

    @property
    def requires_reference(self) -> bool:
        # This evaluator needs a reference answer to score against.
        return True

    @property
    def evaluation_name(self) -> str:
        return "reference_answer_f1"

    def _evaluate_strings(
        self,
        prediction: str,
        reference: Optional[str] = None,
        input: Optional[str] = None,
        **kwargs,
    ) -> tuple[float, str]:
        if not reference:
            return 0.0, "No reference answer provided."

        pred_tokens = set(re.findall(r"\w+", prediction.lower()))
        ref_tokens = set(re.findall(r"\w+", reference.lower()))

        if not ref_tokens:
            return 0.0, "Reference answer has no tokens."
        if not pred_tokens:
            return 0.0, "Prediction has no tokens."

        overlap = pred_tokens & ref_tokens
        precision = len(overlap) / len(pred_tokens)
        recall = len(overlap) / len(ref_tokens)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)

        explanation = (
            f"Token overlap F1={f1:.2f} "
            f"(precision={precision:.2f}, recall={recall:.2f})"
        )
        return f1, explanation


def llm_judge_evaluate(
    prompt: str,
    prediction: str,
    criteria: str = "helpfulness",
    llm: Optional[ChatOpenAI] = None,
) -> dict:
    """
    Use an LLM as a judge to assess response quality based on a given criterion.

    This is a basic demonstration of the LLM-as-judge pattern. It constructs a
    simple instruction for the LLM to rate the response on a scale of 1 to 5,
    providing a score and a short justification.

    Args:
        prompt: The original user prompt that generated the response.
        prediction: The model's response to evaluate.
        criteria: The quality dimension to judge (e.g., "helpfulness", "correctness", "conciseness").
        llm: An optional ChatOpenAI instance. If not provided, a default one is created using
            the OPENAI_API_KEY environment variable.

    Returns:
        A dictionary with keys:
            - "score": integer from 1 to 5 (or None if parsing fails)
            - "explanation": the LLM's textual reasoning
            - "raw_output": the full LLM output for transparency
    """
    if llm is None:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    judge_prompt = f"""You are an impartial judge evaluating the quality of a response.
Given the user prompt and the model's response, rate the response on a scale of 1 (poor) to 5 (excellent) based on {criteria}.

User prompt: {prompt}

Model response: {prediction}

Return your evaluation in the following format:
Score: <integer 1-5>
Explanation: <brief reason for the score>"""

    response = llm.invoke(judge_prompt)
    raw_output = response.content if hasattr(response, "content") else str(response)

    # Parse score and explanation
    score = None
    explanation = ""
    lines = raw_output.split("\n")
    for line in lines:
        if line.lower().startswith("score:"):
            try:
                score = int(line.split(":", 1)[1].strip())
            except ValueError:
                score = None
        elif line.lower().startswith("explanation:"):
            explanation = line.split(":", 1)[1].strip()
    if not explanation:
        # Fallback: use the whole output as explanation
        explanation = raw_output

    return {
        "score": score,
        "explanation": explanation,
        "raw_output": raw_output,
    }


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
        custom_evaluator = KeywordAndLengthEvaluator(keywords=keywords, min_length=min_length)
        results["custom_result"] = custom_evaluator.evaluate_strings(prediction=prediction)

    return results


# Example usage demonstrating the LLM-as-judge evaluation
if __name__ == "__main__":
    # Ensure an OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable.")
        exit(1)

    # Create a sample prompt and response
    sample_prompt = "What are the benefits of regular exercise?"
    sample_response = (
        "Regular exercise improves cardiovascular health, strengthens muscles, "
        "boosts mental well-being, and helps maintain a healthy weight. "
        "It also reduces the risk of chronic diseases like diabetes and hypertension."
    )

    # Use LLM-as-judge to assess helpfulness
    judge_result = llm_judge_evaluate(
        prompt=sample_prompt,
        prediction=sample_response,
        criteria="helpfulness"
    )
    print("LLM-as-judge evaluation:")
    print(f"  Score: {judge_result['score']}/5")
    print(f"  Explanation: {judge_result['explanation']}")
    print()

    # Also demonstrate the criteria evaluator (if desired)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    criteria_result = load_evaluator("criteria", criteria="relevance", llm=llm).evaluate_strings(
        prediction=sample_response,
        input=sample_prompt
    )
    print("Criteria-based evaluation (relevance):")
    print(f"  Score: {criteria_result['score']}")
    print(f"  Reasoning: {criteria_result['reasoning']}")
