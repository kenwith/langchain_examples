"""Tests for the ollama_async_llm_as_judge example.

This test suite verifies that the example module can be imported and that
its pure helper functions behave as expected. No Ollama server is required.

| Test Function | Description |
|---------------|-------------|
| test_module_import | Verifies the example module can be imported. |
| test_build_judge_prompt | Verifies prompt construction includes question, answer, and rubric. |
| test_parse_judge_response | Verifies numeric score extraction from judge responses. |
"""

import os

import ollama_async_llm_as_judge as judge


def _get_model():
    """Return an Ollama chat model using the provider-agnostic initializer."""
    from langchain.chat_models import init_chat_model

    return init_chat_model(
        model=os.getenv("OLLAMA_MODEL", "llama3"),
        provider="ollama",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )


def test_module_import():
    """The example module should be importable."""
    assert judge is not None


def test_build_judge_prompt():
    """build_judge_prompt should produce a prompt containing the question and answer."""
    prompt = judge.build_judge_prompt(
        question="What is the capital of France?",
        answer="Paris",
        rubric="Be concise and accurate.",
    )
    assert "What is the capital of France?" in prompt
    assert "Paris" in prompt
    assert "Be concise and accurate." in prompt


def test_parse_judge_response():
    """parse_judge_response should extract a numeric score from a response."""
    assert judge.parse_judge_response("Score: 5") == 5
    assert judge.parse_judge_response("5 out of 10") == 5
    assert judge.parse_judge_response("Rating: 7/10") == 7


if __name__ == "__main__":
    test_module_import()
    test_build_judge_prompt()
    test_parse_judge_response()
    print("All tests passed.")
