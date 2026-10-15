"""78. Async Agent with Memory

This example demonstrates how to build an asynchronous agent that maintains
conversation memory across turns using LangGraph's checkpointer and uses
tool calling to perform tasks.

The model is initialized provider-agnostically via `init_chat_model`, so the
same code works with OpenAI, Anthropic, or other supported providers by
setting the appropriate environment variables.
"""

import asyncio
import os
from datetime import datetime

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent


@tool
async def get_current_time() -> str:
    """Returns the current UTC time as an ISO 8601 string."""
    return datetime.utcnow().isoformat()


def create_agent():
    """Create a ReAct agent with tool calling and checkpointer memory."""
    model = init_chat_model(
        model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
        provider=os.getenv("CHAT_MODEL_PROVIDER", "openai"),
        temperature=0,
    )
    tools = [get_current_time]
    checkpointer = MemorySaver()
    agent = create_react_agent(model, tools, checkpointer=checkpointer)
    return agent


async def main() -> None:
    """Run a multi-turn conversation to demonstrate memory."""
    agent = create_agent()
    config = {"configurable": {"thread_id": "demo-thread"}}

    # First turn: ask for the current time
    response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "What time is it?"}]},
        config,
    )
    print("Assistant:", response["messages"][-1].content)

    # Second turn: ask about the previous question to verify memory
    response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "What was my first question?"}]},
        config,
    )
    print("Assistant:", response["messages"][-1].content)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Ensure the required API key is set (e.g., OPENAI_API_KEY).")
