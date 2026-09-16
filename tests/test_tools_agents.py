"""Tests for tools and agents.

This module contains tests that verify the agent can call tools and produce a final answer.
It uses fake model responses to simulate the LLM behavior.

| Test | Description |
|------|-------------|
| test_agent_calls_tool_and_produces_final_answer | Verifies the agent can call a tool and produce a final answer |
"""

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.language_models.fake_chat_models import FakeChatModel
from langchain_core.messages import AIMessage


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def test_agent_calls_tool_and_produces_final_answer():
    """Test that an agent can call a tool and produce a final answer."""
    # Create a fake model that returns a tool call first, then a final answer
    fake_model = FakeChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "add",
                        "args": {"a": 2, "b": 3},
                        "id": "call_1",
                    }
                ],
            ),
            AIMessage(content="The sum of 2 and 3 is 5."),
        ]
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    agent = create_tool_calling_agent(llm=fake_model, tools=[add], prompt=prompt)
    executor = AgentExecutor(agent=agent, tools=[add], verbose=False)

    result = executor.invoke({"input": "What is 2 + 3?"})

    assert "5" in result["output"]
    assert "sum" in result["output"].lower()


if __name__ == "__main__":
    test_agent_calls_tool_and_produces_final_answer()
    print("Test passed.")
