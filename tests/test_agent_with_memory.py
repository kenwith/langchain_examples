"""Test cases for agent with memory.

| Test | Description |
|------|-------------|
| test_agent_responds_correctly | Verifies that the agent returns a valid response |
| test_agent_retains_memory | Verifies that the agent retains conversation history across calls |
"""

import os
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import init_chat_model
from langchain.memory import ConversationBufferMemory
from langchain_core.language_models.fake_chat_models import FakeListChatModel


def test_agent_responds_correctly():
    """Verify that the agent produces a response."""
    llm = FakeListChatModel(responses=["Hello, how can I help?"])
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent = initialize_agent(
        tools=[],
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
    )
    response = agent.invoke({"input": "Hi"})
    assert response["output"] == "Hello, how can I help?"


def test_agent_retains_memory():
    """Verify that the agent remembers previous interactions."""
    llm = FakeListChatModel(responses=["Hello!", "How can I assist you?"])
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent = initialize_agent(
        tools=[],
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
    )
    agent.invoke({"input": "My name is Alice"})
    agent.invoke({"input": "What's my name?"})

    # Expect 4 messages: user1, assistant1, user2, assistant2
    assert len(memory.buffer) == 4
    assert memory.buffer[0].content == "My name is Alice"
    assert memory.buffer[2].content == "What's my name?"


def demo():
    """Run a quick demonstration if an API key is available."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set. Skipping demo.")
        return

    llm = init_chat_model("gpt-3.5-turbo", model_provider="openai")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent = initialize_agent(
        tools=[],
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
    )

    response = agent.invoke({"input": "Hi, my name is Bob"})
    print(f"Assistant: {response['output']}")

    response = agent.invoke({"input": "What is my name?"})
    print(f"Assistant: {response['output']}")


if __name__ == "__main__":
    demo()
