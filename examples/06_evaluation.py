"""Example: Custom evaluator for required keywords.

This script demonstrates how to define a custom evaluator that checks whether
the model's response contains a set of required keywords, and how to include
that evaluator in an evaluation run using LangChain's evaluation framework.
"""

from __future__ import annotations

import os
from typing import List, Optional

from langchain.chat_models import init_chat_model
from langchain.evaluation import EvaluatorType
from langchain.evaluation.schema import StringEvaluator
from langchain.smith import RunEvalConfig
from langsmith import Client
from langsmith.evaluation import evaluate


class RequiredKeywordsEvaluator(StringEvaluator):
    """Evaluator that verifies the presence of required keywords.

    The evaluator checks if all required keywords appear in the prediction.
    It returns a score of 1.0 if all keywords are found, and 0.0 otherwise.
    """

    def __init__(self, required_keywords: List[str]):
        """Initialize with the list of keywords to check for.

        Args:
            required_keywords: List of keywords that must be present.
        """
        self.required_keywords = required_keywords

    @property
    def evaluation_name(self) -> str:
        """Return the name of the evaluator."""
        return "required_keywords"

    def _evaluate_strings(
        self,
        prediction: str,
        reference: Optional[str] = None,
        input: Optional[str] = None,
        **kwargs,
    ) -> dict:
        """Evaluate whether the prediction contains all required keywords.

        Args:
            prediction: The model's generated response.
            reference: The reference answer (optional).
            input: The original input (optional).
            **kwargs: Additional keyword arguments.

        Returns:
            A dictionary with the score and the list of missing keywords.
        """
        missing_keywords = [
            keyword
            for keyword in self.required_keywords
            if keyword.lower() not in prediction.lower()
        ]
        score = 1.0 if not missing_keywords else 0.0
        return {
            "score": score,
            "missing_keywords": missing_keywords,
        }


def build_evaluator(required_keywords: List[str]) -> RunEvalConfig:
    """Build the evaluation configuration.

    This helper creates a custom keyword evaluator and a provider-agnostic
    chat model for the built-in QA evaluator. The model is configured through
    the ``EVAL_MODEL`` and ``EVAL_MODEL_PROVIDER`` environment variables,
    making the example usable with any LangChain-supported chat model provider.

    Args:
        required_keywords: List of keywords that must be present in responses.

    Returns:
        A ``RunEvalConfig`` containing both the built-in QA evaluator and the
        custom keyword evaluator.
    """
    # Create the custom evaluator.
    custom_evaluator = RequiredKeywordsEvaluator(required_keywords)

    # Initialize a provider-agnostic chat model for the built-in evaluator.
    # Configure the model and provider through environment variables.
    eval_llm = init_chat_model(
        model=os.getenv("EVAL_MODEL", "gpt-4o"),
        model_provider=os.getenv("EVAL_MODEL_PROVIDER", "openai"),
    )

    # Configure the evaluation run to include both a standard evaluator
    # and the custom keyword evaluator.
    eval_config = RunEvalConfig(
        evaluators=[
            EvaluatorType.QA,
            custom_evaluator,
        ],
        eval_llm=eval_llm,
    )
    return eval_config


def run_evaluation() -> "EvaluationResults":
    """Run the evaluation and return the results."""
    # Define the required keywords for the task.
    required_keywords = ["langchain", "evaluation"]

    # Build the evaluation configuration using the helper.
    eval_config = build_evaluator(required_keywords)

    # Initialize the LangSmith client and run the evaluation.
    client = Client()
    results = evaluate(
        "my_dataset",
        data=client.list_dataset_runs("my_dataset"),
        evaluators=eval_config,
    )
    return results


def format_results(results) -> None:
    """Print evaluation results in a readable format."""
    print("Evaluation Results")
    print("------------------")
    for res in results.results:
        print(f"{res.key}: {res.score}")

        # Show additional details when available.
        metadata = getattr(res, "metadata", None) or {}
        missing_keywords = metadata.get("missing_keywords")
        if missing_keywords:
            print(f"  Missing keywords: {', '.join(missing_keywords)}")

        comment = getattr(res, "comment", None)
        if comment:
            print(f"  Comment: {comment}")
    print("------------------")


def main() -> None:
    """Run the evaluation and display the results."""
    results = run_evaluation()
    format_results(results)


if __name__ == "__main__":
    main()
