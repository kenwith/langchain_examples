"""
Tools & Agents Example

Demonstrates: Function calling, ReAct agent, structured tool outputs
Provider-agnostic using init_chat_model
"""
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

load_dotenv()


def get_model():
    model_name = os.getenv("LANGCHAIN_MODEL", "openai/gpt-4o-mini")
    return init_chat_model(model_name)


# =============================================================================
# Tool Definitions
# =============================================================================

@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculate(expression: str) -> float:
    """Evaluate a mathematical expression safely."""
    # Safe evaluation - only allow basic math
    allowed_names = {"__builtins__": {}}
    try:
        result = eval(expression, allowed_names)
        return float(result)
    except Exception as e:
        return f"Error: {e}"


@tool
def search_knowledge_base(query: str) -> str:
    """Search a mock knowledge base for information."""
    knowledge = {
        "langchain": "LangChain is a framework for building LLM applications.",
        "langgraph": "LangGraph enables stateful multi-agent workflows.",
        "rag": "RAG combines retrieval with generation for accurate answers.",
        "vector store": "Vector stores enable semantic search via embeddings.",
    }
    query_lower = query.lower()
    for key, value in knowledge.items():
        if key in query_lower:
            return value
    return "No information found for that query."


class WeatherInput(BaseModel):
    """Input for weather tool."""
    location: str = Field(description="City and state, e.g., 'San Francisco, CA'")
    unit: str = Field(default="celsius", description="Temperature unit: celsius or fahrenheit")


@tool(args_schema=WeatherInput)
def get_weather(location: str, unit: str = "celsius") -> str:
    """Get mock weather for a location."""
    # Mock data
    mock_weather = {
        "san francisco": {"temp_c": 18, "condition": "foggy"},
        "new york": {"temp_c": 22, "condition": "sunny"},
        "london": {"temp_c": 15, "condition": "rainy"},
    }
    key = location.lower().split(",")[0].strip()
    if key in mock_weather:
        data = mock_weather[key]
        temp = data["temp_c"]
        if unit == "fahrenheit":
            temp = temp * 9/5 + 32
            unit_sym = "°F"
        else:
            unit_sym = "°C"
        return f"{location}: {temp}{unit_sym}, {data['condition']}"
    return f"No weather data for {location}"


# =============================================================================
# Agent Examples
# =============================================================================

def tool_calling_agent():
    """Using create_tool_calling_agent (LangChain native)"""
    print("=== Tool Calling Agent ===")

    model = get_model()
    tools = [get_current_time, calculate, search_knowledge_base, get_weather]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant with access to tools. Use them when needed."),
        ("user", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(model, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    queries = [
        "What time is it?",
        "Calculate 15 * 23 + 45",
        "What is LangGraph?",
        "What's the weather in San Francisco?",
    ]

    for q in queries:
        print(f"\nQ: {q}")
        result = executor.invoke({"input": q})
        print(f"A: {result['output']}")


def react_agent_langgraph():
    """Using LangGraph's create_react_agent (ReAct pattern)"""
    print("\n=== ReAct Agent (LangGraph) ===")

    model = get_model()
    tools = [get_current_time, calculate, search_knowledge_base, get_weather]

    agent = create_react_agent(model, tools)

    queries = [
        "What time is it?",
        "Calculate 100 / 4 * 5",
        "Tell me about RAG",
        "Weather in London in fahrenheit",
    ]

    for q in queries:
        print(f"\nQ: {q}")
        result = agent.invoke({"messages": [("user", q)]})
        # Get the last message (AI response)
        for msg in result["messages"]:
            if msg.type == "ai":
                print(f"A: {msg.content}")


def structured_tool_agent():
    """Agent with structured output tools"""
    print("\n=== Structured Tool Agent ===")

    class TaskResult(BaseModel):
        task: str
        status: str  # completed, failed, in_progress
        details: str

    @tool(args_schema=TaskResult)
    def create_task(task: str, status: str, details: str) -> TaskResult:
        """Create a structured task result."""
        return TaskResult(task=task, status=status, details=details)

    model = get_model()
    tools = [create_task]

    agent = create_react_agent(model, tools)

    result = agent.invoke({
        "messages": [("user", "Create a task for 'review PR #42' with status 'in_progress' and details 'Waiting for CI'")]
    })

    for msg in result["messages"]:
        if msg.type == "ai":
            print(f"A: {msg.content}")

    # Check for tool calls
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"Tool call: {tc}")


def parallel_tool_calls():
    """Agent making parallel tool calls"""
    print("\n=== Parallel Tool Calls ===")

    model = get_model()
    tools = [get_current_time, calculate, get_weather]

    agent = create_react_agent(model, tools)

    # This should trigger parallel calls
    result = agent.invoke({
        "messages": [("user", "What time is it, and calculate 25 * 4, and get weather for New York?")]
    })

    for msg in result["messages"]:
        if msg.type == "ai":
            print(f"A: {msg.content}")


if __name__ == "__main__":
    tool_calling_agent()
    react_agent_langgraph()
    structured_tool_agent()
    parallel_tool_calls()
    print("\nAll tools/agents examples completed!")