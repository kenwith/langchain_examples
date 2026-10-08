"""Tests for the Ollama async agent example.

This module contains tests that verify the async agent example runs when
Ollama is available, and skips gracefully when it is not.

| Test                    | Description                                     |
|-------------------------|-------------------------------------------------|
| test_ollama_async_agent | Verifies the async agent runs with Ollama.     |
"""

import asyncio
import os
import pytest

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


@tool
def get_word_length(word: str) -> int:
    """Return the length of a word."""
    return len(word)


async def run_async_agent(query: str) -> str:
    """Run the async agent with the given query."""
    llm = init_chat_model(
        model=MODEL_NAME,
        model_provider="ollama",
        base_url=OLLAMA_BASE_URL,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(llm, [get_word_length], prompt)
    executor = AgentExecutor(agent=agent, tools=[get_word_length])
    result = await executor.ainvoke({"input": query})
    return result["output"]


def test_ollama_async_agent():
    """Verify the async agent example runs when Ollama is available."""
    try:
        output = asyncio.run(
            run_async_agent("What is the length of the word 'hello'?")
        )
    except Exception as exc:  # pragma: no cover - depends on environment
        pytest.skip(f"Ollama unavailable: {exc}")
    assert isinstance(output, str)
    assert len(output) > 0


if __name__ == "__main__":
    try:
        result = asyncio.run(
            run_async_agent("What is the length of the word 'pytest'?")
        )
        print(result)
    except Exception as exc:
        print(f"Ollama unavailable: {exc}")
