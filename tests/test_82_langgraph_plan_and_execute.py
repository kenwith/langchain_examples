"""# Test: Plan and Execute

| Test | Description |
|------|-------------|
| test_plan_and_execute | Runs the plan-and-execute graph and verifies the final answer contains expected content. |
"""

import os

import pytest
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import PlanAndExecute


def test_plan_and_execute():
    """Run the plan-and-execute graph and verify the final answer contains expected content."""
    model = init_chat_model(os.getenv("CHAT_MODEL", "gpt-4o"), temperature=0)
    agent = PlanAndExecute(model, tools=[])
    result = agent.invoke({"input": "What is the capital of France?"})
    assert "Paris" in result["response"]


if __name__ == "__main__":
    test_plan_and_execute()
    print("Test passed.")
