"""
Example 56: Ollama Evaluation
Uses a local judge model to evaluate generated answers.
"""

import os

from langchain.chat_models import init_chat_model


def generate_answer(question: str, model_name: str = "llama3.1", base_url: str = "http://localhost:11434") -> str:
    """Generate an answer to the given question using an Ollama model."""
    llm = init_chat_model(model_name, provider="ollama", base_url=base_url)
    response = llm.invoke(question)
    return response.content


def evaluate_answer(question: str, answer: str, judge_model_name: str = "llama3.1", base_url: str = "http://localhost:11434") -> str:
    """Evaluate the quality of an answer using a local judge model."""
    judge_prompt = f"""
You are an impartial judge. Evaluate the following answer to the question.
Provide a score from 1 to 10 (10 being excellent) and a brief justification.

Question: {question}
Answer: {answer}

Your evaluation (format as "Score: X/10\nJustification: ..."):
"""
    judge_llm = init_chat_model(judge_model_name, provider="ollama", base_url=base_url)
    response = judge_llm.invoke(judge_prompt)
    return response.content


def main() -> None:
    """Run a simple demonstration of generation and evaluation."""
    question = "What is the capital of France?"
    print(f"Question: {question}\n")

    # Generate an answer
    answer = generate_answer(question)
    print(f"Generated answer: {answer}\n")

    # Evaluate the answer
    evaluation = evaluate_answer(question, answer)
    print(f"Judge evaluation:\n{evaluation}")


if __name__ == "__main__":
    main()
