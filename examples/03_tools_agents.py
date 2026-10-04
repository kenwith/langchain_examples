"""
Example: Tools and Agents with LangChain
This script demonstrates how to create an agent with tools using LangChain.
It includes a factory function to create agents, a helper to run queries,
and documentation on adding custom tools.
"""

import os
from typing import List, Optional

from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.tools import Tool, tool
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

# Set your OpenAI API key here or in environment variables
os.environ.setdefault("OPENAI_API_KEY", "your-api-key")


# ==================== Tool Definitions ====================

@tool
def add_numbers(a: str, b: str) -> str:
    """
    Add two numbers together and return the result.

    Args:
        a (str): The first number as a string.
        b (str): The second number as a string.

    Returns:
        str: The sum of a and b as a string.
    """
    try:
        result = float(a) + float(b)
        return str(result)
    except ValueError:
        return "Error: Both inputs must be numeric."


@tool
def multiply_numbers(a: str, b: str) -> str:
    """
    Multiply two numbers and return the product.

    Args:
        a (str): The first number as a string.
        b (str): The second number as a string.

    Returns:
        str: The product of a and b as a string.
    """
    try:
        result = float(a) * float(b)
        return str(result)
    except ValueError:
        return "Error: Both inputs must be numeric."


@tool
def get_weather(city: str) -> str:
    """
    Get the current weather for a given city.

    This tool simulates a weather API. In a real implementation, you would
    call an external service. For demonstration, it returns a fixed string.

    Args:
        city (str): The name of the city.

    Returns:
        str: A weather report for the city.
    """
    # Simulated weather data
    weather_data = {
        "new york": "Sunny, 72°F",
        "london": "Cloudy, 60°F",
        "tokyo": "Rainy, 65°F",
    }
    city_lower = city.lower()
    if city_lower in weather_data:
        return f"Weather in {city}: {weather_data[city_lower]}"
    else:
        return f"Weather data not available for {city}."


@tool
def search_web(query: str) -> str:
    """
    Search the web for information.

    This tool simulates a web search. In a real implementation, you would
    use a search API like Google or Bing. For demonstration, it returns a
    static response.

    Args:
        query (str): The search query.

    Returns:
        str: A summary of search results.
    """
    # Simulated search result
    return f"Top result for '{query}': This is a simulated web search result."


# ==================== Agent Factory ====================

def create_agent(
    tools: List[Tool],
    llm: Runnable,
    system_prompt: Optional[str] = None,
    verbose: bool = True,
) -> AgentExecutor:
    """
    Create a ReAct agent executor with the given tools and language model.

    This factory function simplifies the process of setting up an agent by
    handling the prompt template, agent creation, and executor configuration.

    Args:
        tools (List[Tool]): List of tools available to the agent.
        llm (Runnable): The language model to use.
        system_prompt (Optional[str]): Custom system prompt. If None, a default
            prompt is used.
        verbose (bool): Whether to print verbose output.

    Returns:
        AgentExecutor: Configured agent executor ready to be invoked.
    """
    # Define a default system prompt if none provided
    if system_prompt is None:
        system_prompt = (
            "You are a helpful assistant. Use the provided tools to answer "
            "user questions. Always use a tool when necessary, and be precise "
            "with the tool inputs."
        )

    # Create the prompt template for the ReAct agent
    prompt = PromptTemplate.from_template(
        """{system_prompt}

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""
    )

    # Create the agent
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
        output_parser=ReActSingleInputOutputParser(),
    )

    # Create and return the executor
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        handle_parsing_errors=True,
    )
    return executor


# ==================== Run Agent Helper ====================

def run_agent(agent_executor: AgentExecutor, query: str) -> str:
    """
    Execute a single query against the agent and return the answer.

    This helper encapsulates the invocation logic, making it easy to run
    multiple queries or integrate the agent into a larger application.

    Args:
        agent_executor (AgentExecutor): The configured agent executor.
        query (str): The user's question or instruction.

    Returns:
        str: The agent's final answer.
    """
    print(f"\n--- Query: {query} ---")
    response = agent_executor.invoke({"input": query})
    answer = response["output"]
    print(f"Answer: {answer}")
    return answer


# ==================== Main Execution ====================

if __name__ == "__main__":
    # Define the tools list with expanded descriptions
    # To add a new tool:
    # 1. Define a function with @tool decorator (or use Tool class).
    # 2. Ensure the function has a clear docstring that describes what it does
    #    and the arguments it expects.
    # 3. Add the function to the `tools` list below.
    tools = [
        add_numbers,
        multiply_numbers,
        get_weather,
        search_web,
    ]

    # Initialize the language model
    llm = ChatOpenAI(model="gpt-4", temperature=0)

    # Create the agent using the factory function
    agent_executor = create_agent(
        tools=tools,
        llm=llm,
        verbose=True,
        system_prompt=(
            "You are an assistant that can perform arithmetic, check weather, "
            "and search the web. Use the appropriate tool for each task."
        ),
    )

    # Example queries
    queries = [
        "What is 123 + 456?",
        "What is the weather in London?",
        "Search for LangChain documentation.",
        "Multiply 7.5 by 3.",
    ]

    # Run the agent on each query using the helper
    for query in queries:
        run_agent(agent_executor, query)
