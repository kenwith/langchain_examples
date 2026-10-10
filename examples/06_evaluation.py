"""Example: Custom evaluator for required keywords.

This script demonstrates how to define a custom evaluator that checks whether
the model's response contains a set of required keywords, and how to include
that evaluator in an evaluation run using LangChain's evaluation framework.
"""

from __future__ import annotations

from typing import List, Optional

from langchain.evaluation import EvaluatorType
from langchain.evaluation.schema import StringEvaluator
from langchain.smith import RunEvalConfig
from langchain.llms import OpenAI
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


def main() -> None:
    """Run the evaluation with the custom keyword evaluator."""
    # Define the required keywords for the task.
    required_keywords = ["langchain", "evaluation"]

    # Create the custom evaluator.
    custom_evaluator = RequiredKeywordsEvaluator(required_keywords)

    # Configure the evaluation run to include both a standard evaluator
    # and the custom keyword evaluator.
    eval_config = RunEvalConfig(
        evaluators=[
            EvaluatorType.QA,
            custom_evaluator,
        ]
    )

    # Initialize the LangSmith client and run the evaluation.
    client = Client()
    evaluate(
        "my_dataset",
        data=client.list_dataset_runs("my_dataset"),
        evaluators=eval_config,
    )


if __name__ == "__main__":
    main()
