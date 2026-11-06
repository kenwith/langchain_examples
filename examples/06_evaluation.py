"""Example: Custom evaluator for a simple accuracy metric.

This script demonstrates how to define a custom evaluator that computes a
simple accuracy metric by comparing the model's prediction to a reference
answer, and how to include that evaluator in an evaluation run using
LangChain's evaluation framework.

The evaluation run uses the following evaluator:

- ``AccuracyEvaluator``: A custom evaluator that returns a score of 1.0 if
  the prediction exactly matches the reference answer (ignoring case and
  surrounding whitespace) and 0.0 otherwise.
"""

from __future__ import annotations

from typing import Optional

from langchain.evaluation.schema import StringEvaluator
from langchain.smith import RunEvalConfig
from langsmith import Client
from langsmith.evaluation import evaluate


class AccuracyEvaluator(StringEvaluator):
    """Evaluator that computes a simple accuracy metric.

    The evaluator compares the prediction to a reference answer and returns
    a score of 1.0 if they match exactly (after stripping whitespace and
    ignoring case) and 0.0 otherwise.
    """

    @property
    def evaluation_name(self) -> str:
        """Return the name of the evaluator."""
        return "accuracy"

    def _evaluate_strings(
        self,
        prediction: str,
        reference: Optional[str] = None,
        input: Optional[str] = None,
        **kwargs,
    ) -> dict:
        """Evaluate the accuracy of the prediction against the reference.

        Args:
            prediction: The model's generated response.
            reference: The reference answer.
            input: The original input (optional).
            **kwargs: Additional keyword arguments.

        Returns:
            A dictionary with the score and a comment if no reference is provided.
        """
        if reference is None:
            return {
                "score": 0.0,
                "comment": "No reference answer provided.",
            }

        normalized_prediction = prediction.strip().lower()
        normalized_reference = reference.strip().lower()
        score = 1.0 if normalized_prediction == normalized_reference else 0.0
        return {"score": score}


def build_evaluator() -> RunEvalConfig:
    """Build the evaluation configuration.

    This helper creates a custom accuracy evaluator and returns a
    ``RunEvalConfig`` that includes it in the evaluation run.

    Returns:
        A ``RunEvalConfig`` containing the custom accuracy evaluator.
    """
    return RunEvalConfig(evaluators=[AccuracyEvaluator()])


def run_evaluation() -> "EvaluationResults":
    """Run the evaluation and return the results."""
    eval_config = build_evaluator()

    client = Client()
    results = evaluate(
        "my_dataset",
        data=client.list_dataset_runs("my_dataset"),
        evaluators=eval_config,
    )
    return results


def _evaluator_description(key: str) -> str:
    """Return a human-readable description for a given evaluator key."""
    descriptions = {
        "accuracy": "Checks whether the prediction exactly matches the reference answer.",
    }
    return descriptions.get(key, "No description available.")


def format_results(results) -> None:
    """Print evaluation results in a readable format with clear metric names."""
    print("Evaluation Results")
    print("------------------")

    # Track pass/fail counts for a summary.
    passed = 0
    failed = 0

    for res in results.results:
        evaluator_name = res.key
        score = res.score

        # Print a clear header for each evaluator.
        print(f"\nEvaluator: {evaluator_name}")
        print(f"  Description: {_evaluator_description(evaluator_name)}")

        # Print the score with an explanation.
        if score == 1.0:
            print(f"  Score: 1.0 (pass)")
            passed += 1
        elif score == 0.0:
            print(f"  Score: 0.0 (fail)")
            failed += 1
        else:
            print(f"  Score: {score:.2f}")
            # For non-binary scores, consider a threshold for pass/fail.
            if score >= 0.5:
                passed += 1
            else:
                failed += 1

        comment = getattr(res, "comment", None)
        if comment:
            print(f"  Comment: {comment}")

    # Print a summary of the evaluation run.
    print("\n------------------")
    print(f"Summary: {passed} passed, {failed} failed out of {len(results.results)} evaluators.")
    print("------------------")


def main() -> None:
    """Run the evaluation and display the results."""
    results = run_evaluation()
    format_results(results)


if __name__ == "__main__":
    main()
