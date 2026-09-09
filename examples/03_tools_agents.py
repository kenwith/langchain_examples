"""
Tools & Agents Example

Demonstrates: Function calling, ReAct agent, structured tool outputs
Provider-agnostic using init_chat_model
"""
import os
import json
from datetime import datetime
from typing import List, Optional
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

load_dotenv()


def get_model():
    """Initialize chat model based on LANGCHAIN_MODEL env var.
    Throws a helpful error if the API key for the specified provider is missing.
    """
    model_name = os.getenv("LANGCHAIN_MODEL", "openai/gpt-4o-mini")
    # Known provider prefixes and their required API key environment variables
    provider_key_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "cohere": "COHERE_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "azure": "AZURE_OPENAI_API_KEY",
        "together": "TOGETHER_API_KEY",
        "fireworks": "FIREWORKS_API_KEY",
        "groq": "GROQ_API_KEY",
    }
    provider = model_name.split("/")[0].lower()
    if provider in provider_key_map:
        key_var = provider_key_map[provider]
        if not os.getenv(key_var):
            raise EnvironmentError(
                f"Missing API key for provider '{provider}'. "
                f"Set the {key_var} environment variable (e.g., in your .env file)."
            )
    return init_chat_model(model_name)


# =============================================================================
# Tool Definitions
# =============================================================================

@tool
def get_current_time() -> str:
    """Get the current date and time.

    Returns:
        Current timestamp in the format 'YYYY-MM-DD HH:MM:SS'.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class CalculateInput(BaseModel):
    """Input for the safe calculator tool."""
    expression: str = Field(description="The mathematical expression to evaluate, e.g., '15 * 23'.")


@tool(args_schema=CalculateInput)
def calculate(expression: str) -> float:
    """Evaluate a mathematical expression safely.

    Supports basic arithmetic operators (+, -, *, /, parentheses) and numbers.
    Does not allow access to Python built‑ins or modules.

    Args:
        expression: A string containing a valid mathematical expression.

    Returns:
        The result as a float, or a string error message if evaluation fails.
    """
    allowed_names = {"__builtins__": {}}
    try:
        result = eval(expression, allowed_names)
        return float(result)
    except Exception as e:
        return f"Error: {e}"


class SearchInput(BaseModel):
    """Input for the knowledge base search tool."""
    query: str = Field(description="The search query to look up in the knowledge base.")


@tool(args_schema=SearchInput)
def search_knowledge_base(query: str) -> str:
    """Search a mock knowledge base for information.

    Contains a static collection of entries about topics like LangChain, LangGraph, RAG, and more.

    Args:
        query: The user's search query.

    Returns:
        A short answer if the query matches an entry, otherwise a fallback message.
    """
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
    """Input for the weather tool."""
    location: str = Field(description="City and state, e.g., 'San Francisco, CA'")
    unit: str = Field(default="celsius", description="Temperature unit: celsius or fahrenheit")


@tool(args_schema=WeatherInput)
def get_weather(location: str, unit: str = "celsius") -> str:
    """Get mock weather data for a location.

    Args:
        location: City name (and optionally state), e.g., 'San Francisco, CA'.
        unit: Either 'celsius' (default) or 'fahrenheit'.

    Returns:
        A string describing the current weather for the location.
    """
    # Mock data - in a real app this would call a live weather API
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
# Pydantic Structured Output Models
# =============================================================================

class WeatherReport(BaseModel):
    """Structured weather report output."""
    location: str = Field(description="Location name")
    temperature: float = Field(description="Temperature value")
    unit: str = Field(description="Temperature unit (celsius/fahrenheit)")
    condition: str = Field(description="Weather condition")
    humidity: Optional[int] = Field(default=None, description="Humidity percentage")
    timestamp: str = Field(description="ISO format timestamp")


class CalculationResult(BaseModel):
    """Structured calculation result."""
    expression: str = Field(description="Original expression")
    result: float = Field(description="Calculated result")
    steps: List[str] = Field(default_factory=list, description="Calculation steps")


class SearchResult(BaseModel):
    """Structured search result."""
    query: str = Field(description="Search query")
    answer: str = Field(description="Found answer")
    source: str = Field(description="Knowledge base source")
    confidence: float = Field(description="Confidence score 0-1")


class MultiToolResult(BaseModel):
    """Container for multiple structured tool results."""
    weather: Optional[WeatherReport] = None
    calculation: Optional[CalculationResult] = None
    search: Optional[SearchResult] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


# =============================================================================
# Structured Output Tools (return Pydantic models)
# =============================================================================

@tool
def get_structured_weather(location: str, unit: str = "celsius") -> WeatherReport:
    """Get a structured weather report for a location.

    Args:
        location: City name (e.g., 'San Francisco, CA').
        unit: 'celsius' (default) or 'fahrenheit'.

    Returns:
        A WeatherReport object with structured weather data.
    """
    mock_weather = {
        "san francisco": {"temp_c": 18, "condition": "foggy", "humidity": 85},
        "new york": {"temp_c": 22, "condition": "sunny", "humidity": 60},
        "london": {"temp_c": 15, "condition": "rainy", "humidity": 90},
    }
    key = location.lower().split(",")[0].strip()
    if key in mock_weather:
        data = mock_weather[key]
        temp = data["temp_c"]
        if unit == "fahrenheit":
            temp = temp * 9/5 + 32
        return WeatherReport(
            location=location,
            temperature=round(temp, 1),
            unit=unit,
            condition=data["condition"],
            humidity=data["humidity"],
            timestamp=datetime.now().isoformat()
        )
    return WeatherReport(
        location=location,
        temperature=0.0,
        unit=unit,
        condition="unknown",
        humidity=None,
        timestamp=datetime.now().isoformat()
    )


@tool
def structured_calculate(expression: str) -> CalculationResult:
    """Evaluate an expression and return a structured result with execution steps.

    Args:
        expression: A string containing a valid mathematical expression.

    Returns:
        A CalculationResult object containing the expression, result, and steps.
    """
    allowed_names = {"__builtins__": {}}
    steps = []
    try:
        steps.append(f"Parsing expression: {expression}")
        result = eval(expression, allowed_names)
        steps.append(f"Evaluated to: {result}")
        return CalculationResult(
            expression=expression,
            result=float(result),
            steps=steps
        )
    except Exception as e:
        return CalculationResult(
            expression=expression,
            result=0.0,
            steps=[f"Error: {e}"]
        )


@tool
def structured_search(query: str) -> SearchResult:
    """Search the knowledge base and return a structured result.

    Args:
        query: The user's search query.

    Returns:
        A SearchResult object with the answer, source, and confidence.
    """
    knowledge = {
        "langchain": ("LangChain is a framework for building LLM applications.", "langchain_docs"),
        "langgraph": ("LangGraph enables stateful multi-agent workflows.", "langgraph_docs"),
        "rag": ("RAG combines retrieval with generation for accurate answers.", "rag_paper"),
        "vector store": ("Vector stores enable semantic search via embeddings.", "vector_db_docs"),
    }
    query_lower = query.lower()
    for key, (value, source) in knowledge.items():
        if key in query_lower:
            return SearchResult(
                query=query,
                answer=value,
                source=source,
                confidence=0.95
            )
    return SearchResult(
        query=query,
        answer="No information found for that query.",
        source="none",
        confidence=0.0
    )


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
    """
    Using LangGraph's create_react_agent (ReAct pattern).

    The ReAct loop works as follows:
      1. The model receives a user message and decides whether to call a tool.
      2. If yes, it emits a Tool Call (the Action).
      3. The tool runs and its result is returned as an Observation.
      4. The model reads the observation and either makes another Tool Call
         or produces a final Answer.
      5. The process repeats until no tool calls are made.
    """
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
    """Agent with structured output tools returning Pydantic models"""
    print("\n=== Structured Tool Agent (Pydantic Output) ===")

    model = get_model()
    tools = [get_structured_weather, structured_calculate, structured_search]

    agent = create_react_agent(model, tools)

    queries = [
        "Get weather for San Francisco in celsius",
        "Calculate 25 * 4 + 10",
        "Search for information about LangGraph",
    ]

    for q in queries:
        print(f"\nQ: {q}")
        result = agent.invoke({"messages": [("user", q)]})
        for msg in result["messages"]:
            if msg.type == "ai":
                print(f"A: {msg.content}")
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"  Tool call: {tc['name']}({tc['args']})")


def parse_structured_outputs():
    """Demonstrate parsing and validating structured tool outputs"""
    print("\n=== Parse & Validate Structured Outputs ===")

    model = get_model()
    tools = [get_structured_weather, structured_calculate, structured_search]

    agent = create_react_agent(model, tools)

    # Single query that triggers multiple tools
    result = agent.invoke({
        "messages": [("user", "Get weather for New York in fahrenheit, calculate 15 * 7, and search for RAG")]
    })

    print("\n--- Raw Messages ---")
    for msg in result["messages"]:
        print(f"{msg.type}: {getattr(msg, 'content', '')}")
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"  Tool Call: {tc['name']} -> {tc}")

    print("\n--- Parsed Structured Results ---")
    # Parse tool messages which contain structured output
    for msg in result["messages"]:
        if msg.type == "tool":
            print(f"\nTool: {msg.name}")
            print(f"Raw content: {msg.content}")
            try:
                # Parse JSON content back to Pydantic model
                parsed = json.loads(msg.content)
                if msg.name == "get_structured_weather":
                    validated = WeatherReport(**parsed)
                    print(f"Validated: {validated.model_dump_json(indent=2)}")
                elif msg.name == "structured_calculate":
                    validated = CalculationResult(**parsed)
                    print(f"Validated: {validated.model_dump_json(indent=2)}")
                elif msg.name == "structured_search":
                    validated = SearchResult(**parsed)
                    print(f"Validated: {validated.model_dump_json(indent=2)}")
            except Exception as e:
                print(f"Parse error: {e}")


def model_with_structured_output():
    """Use model.with_structured_output for direct structured generation"""
    print("\n=== Model.with_structured_output ===")

    model = get_model()

    # Bind structured output to model
    structured_model = model.with_structured_output(MultiToolResult)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Return structured data for the user's request."),
        ("user", "{input}"),
    ])

    chain = prompt | structured_model

    # This will return a validated MultiToolResult directly
    queries = [
        "I need weather for London in celsius, calculate 100 / 4, and search for vector store",
        "Get weather for San Francisco in fahrenheit and calculate 50 * 2",
    ]

    for q in queries:
        print(f"\nQ: {q}")
        try:
            result: MultiToolResult = chain.invoke({"input": q})
            print(f"Structured Result:")
            print(result.model_dump_json(indent=2))
        except Exception as e:
            print(f"Error: {e}")


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
    parse_structured_outputs()
    model_with_structured_output()
    parallel_tool_calls()
    print("\nAll tools/agents examples completed!")
