"""
Evaluation & Testing Example

Demonstrates: Unit testing chains, LLM-as-judge evaluation, LangSmith integration patterns
Provider-agnostic using init_chat_model
"""
import os
import json
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field
import pytest

load_dotenv()


def get_model():
    model_name = os.getenv("LANGCHAIN_MODEL", "openai/gpt-4o-mini")
    return init_chat_model(model_name)


# =============================================================================
# Chain Under Test
# =============================================================================

def create_qa_chain():
    """Create a simple QA chain for testing"""
    model = get_model()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the question concisely. If you don't know, say 'I don't know.'"),
        ("user", "{question}"),
    ])
    return prompt | model | StrOutputParser()


# =============================================================================
# Unit Tests (pytest style)
# =============================================================================

class TestQAChain:
    """Unit tests for QA chain"""

    @pytest.fixture
    def chain(self):
        return create_qa_chain()

    def test_known_fact(self, chain):
        """Test chain answers known fact correctly"""
        result = chain.invoke({"question": "What is the capital of France?"})
        assert "Paris" in result
        assert len(result) < 100  # Concise

    def test_unknown_fact(self, chain):
        """Test chain handles unknown gracefully"""
        result = chain.invoke({"question": "What is the capital of Mars?"})
        assert "don't know" in result.lower() or "unknown" in result.lower()

    def test_empty_question(self, chain):
        """Test chain handles empty input"""
        result = chain.invoke({"question": ""})
        assert isinstance(result, str)
        assert len(result) > 0


# =============================================================================
# LLM-as-Judge Evaluation
# =============================================================================

class EvaluationResult(BaseModel):
    score: int = Field(ge=1, le=5, description="Score 1-5")
    reasoning: str = Field(description="Explanation for score")
    passed: bool = Field(description="Whether test passes (score >= 3)")


def create_evaluator():
    """Create an LLM judge for evaluating responses"""
    model = get_model()

    class EvalOutput(BaseModel):
        score: int = Field(ge=1, le=5)
        reasoning: str
        passed: bool

    parser = JsonOutputParser(pydantic_object=EvalOutput)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an evaluator. Score the response 1-5 based on the criteria.
        Return JSON with: score (1-5), reasoning, passed (true if score >= 3).
        Criteria: {criteria}"""),
        ("user", "Question: {question}\nResponse: {response}\n{format_instructions}"),
    ])

    return prompt | model | parser


def evaluate_responses():
    """Evaluate chain responses using LLM judge"""
    print("=== LLM-as-Judge Evaluation ===")

    chain = create_qa_chain()
    evaluator = create_evaluator()

    test_cases = [
        {
            "question": "What is the capital of France?",
            "criteria": "Accuracy and conciseness. Should be 'Paris'."
        },
        {
            "question": "Explain quantum computing in one sentence.",
            "criteria": "Accuracy, clarity, and single sentence."
        },
        {
            "question": "Who won the 2024 Super Bowl?",
            "criteria": "Accuracy. Should acknowledge knowledge cutoff if unsure."
        },
    ]

    for tc in test_cases:
        print(f"\nQuestion: {tc['question']}")
        response = chain.invoke({"question": tc["question"]})
        print(f"Response: {response}")

        eval_result = evaluator.invoke({
            "question": tc["question"],
            "response": response,
            "criteria": tc["criteria"],
            "format_instructions": evaluator.get_format_instructions()
        })

        print(f"Score: {eval_result['score']}/5")
        print(f"Reasoning: {eval_result['reasoning']}")
        print(f"Passed: {eval_result['passed']}")


# =============================================================================
# Synthetic Data Generation for Testing
# =============================================================================

def generate_test_cases():
    """Generate synthetic test cases using LLM"""
    print("\n=== Synthetic Test Generation ===")

    model = get_model()

    class TestCase(BaseModel):
        question: str
        expected_keywords: list[str]
        difficulty: str

    parser = JsonOutputParser(pydantic_object=list[TestCase])

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Generate 5 diverse test questions for a general knowledge QA system. "
                   "Return JSON array with: question, expected_keywords (list), difficulty (easy/medium/hard)."),
        ("user", "Generate test cases for a QA system about {topic}."),
    ])

    chain = prompt | model | parser

    test_cases = chain.invoke({"topic": "computer science fundamentals"})

    for tc in test_cases:
        print(f"\nQ: {tc['question']}")
        print(f"Keywords: {tc['expected_keywords']}")
        print(f"Difficulty: {tc['difficulty']}")

    return test_cases


# =============================================================================
# Regression Testing Pattern
# =============================================================================

def regression_test_pattern():
    """Pattern for regression testing with stored expected outputs"""
    print("\n=== Regression Testing Pattern ===")

    # In practice, load from JSON file
    golden_dataset = [
        {"input": {"question": "What is 2+2?"}, "expected_contains": "4"},
        {"input": {"question": "Capital of Japan?"}, "expected_contains": "Tokyo"},
        {"input": {"question": "Author of 1984?"}, "expected_contains": "Orwell"},
    ]

    chain = create_qa_chain()
    passed = 0
    failed = 0

    for item in golden_dataset:
        result = chain.invoke(item["input"])
        expected = item["expected_contains"].lower()
        actual = result.lower()

        if expected in actual:
            print(f"✓ PASS: {item['input']['question']}")
            passed += 1
        else:
            print(f"✗ FAIL: {item['input']['question']}")
            print(f"  Expected to contain: {expected}")
            print(f"  Got: {result[:100]}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


# =============================================================================
# LangSmith Integration Pattern (no actual API calls)
# =============================================================================

def langsmith_pattern():
    """Pattern for LangSmith integration (without requiring API key)"""
    print("\n=== LangSmith Integration Pattern ===")

    # This shows the pattern - replace with actual LangSmith calls when configured
    print("""
# LangSmith Integration Pattern:

from langsmith import Client
from langsmith.evaluation import evaluate

client = Client()

# 1. Create dataset
dataset = client.create_dataset("qa-test-cases")
client.create_examples(
    inputs=[{"question": "Capital of France?"}],
    outputs=[{"answer": "Paris"}],
    dataset_id=dataset.id
)

# 2. Define target function
def my_chain(inputs):
    return create_qa_chain().invoke(inputs)

# 3. Define evaluators
def accuracy_evaluator(run, example):
    return {"score": 1 if example.outputs["answer"].lower() in run.outputs["output"].lower() else 0}

# 4. Run evaluation
results = evaluate(
    my_chain,
    data=dataset.name,
    evaluators=[accuracy_evaluator],
    experiment_prefix="qa-eval"
)

print(f"Evaluation results: {results}")
""")


if __name__ == "__main__":
    # Run unit tests
    print("Running unit tests...")
    pytest.main(["-v", __file__ + "::TestQAChain", "--tb=short"])

    # Run evaluations
    evaluate_responses()
    generate_test_cases()
    regression_test_pattern()
    langsmith_pattern()
    print("\nAll evaluation examples completed!")