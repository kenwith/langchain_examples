"""Example 03: Tools and agents with graceful API failure handling.

This example introduces a tool that simulates an API failure and shows how to
catch exceptions inside the tool and return a graceful message to the agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


@dataclass
class WeatherReport:
    """A weather report returned by the weather tool."""
    location: str
    temperature: float
    conditions: str
    error: Optional[str] = None

    def __str__(self) -> str:
        if self.error:
            return f"Could not get weather for {self.location}: {self.error}"
        return f"{self.location}: {self.temperature:.1f}°F, {self.conditions}"


@tool
def get_weather(location: str) -> WeatherReport:
    """Get the current weather for a given location.

    Args:
        location: The city or region to get weather for.

    Returns:
        A WeatherReport with current conditions or an error message.
    """
    # Simulate an external API that can fail.
    # In real code this would be an HTTP request to a weather service.
    unavailable_locations = {"nowhere", "atlantis", "middle-earth"}

    try:
        if location.strip().lower() in unavailable_locations:
            raise ConnectionError(
                f"Weather API is temporarily unreachable for '{location}'"
            )

        # Simulate a successful response.
        return WeatherReport(
            location=location,
            temperature=72.5,
            conditions="Sunny",
        )
    except Exception as exc:
        # Catch the exception inside the tool and return a graceful message.
        return WeatherReport(
            location=location,
            temperature=0.0,
            conditions="unknown",
            error=str(exc),
        )


def run_agent_without_executor() -> None:
    """Run a simple agent loop using a chat model and the weather tool."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_with_tools = llm.bind_tools([get_weather])

    messages = [
        HumanMessage(
            content="What is the weather in Nowhere? Also, what about Seattle?"
        )
    ]

    for step in range(5):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            print("Final answer:", response.content)
            return

        for tool_call in response.tool_calls:
            print(f"\nCalling tool '{tool_call['name']}' with args {tool_call['args']}")
            try:
                tool_result = get_weather.invoke(tool_call["args"])
                print("Tool result:", tool_result)
            except Exception as exc:
                # This should not happen because the tool catches its own exceptions,
                # but we keep a safety net for unexpected errors.
                tool_result = f"Tool error: {exc}"
                print("Tool raised an exception:", exc)
            messages.append(
                ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"])
            )

    print("Agent did not produce a final answer in time.")


def run_agent_with_executor() -> None:
    """Run the agent using LangChain's AgentExecutor."""
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Use tools when needed."),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    agent = create_tool_calling_agent(
        ChatOpenAI(model="gpt-4o-mini", temperature=0),
        [get_weather],
        prompt,
    )
    executor = AgentExecutor(agent=agent, tools=[get_weather], verbose=True)

    result = executor.invoke(
        {"input": "What is the weather in Nowhere? Also, what about Seattle?"}
    )
    print("Agent output:", result["output"])


if __name__ == "__main__":
    # Uncomment one of the following:
    # run_agent_without_executor()
    run_agent_with_executor()
