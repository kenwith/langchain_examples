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
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI

# Load the API key from the environment. Do not hardcode secrets.
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")


# ==================== Tool Definitions ====================

@tool
def calculator(operation: str, a: str, b: str) -> str:
    """Perform a basic arithmetic operation on two numeric strings.

    Supported operations: 'add', 'subtract', 'multiply', 'divide'.
    Returns the result as a string, or an error message if the inputs are
    invalid or the operation is not supported.
    """
    try:
        num_a = float(a)
        num_b = float(b)
    except ValueError:
        return "Error: Both operands must be numeric."

    op = operation.lower()
    if op == "add":
        result = num_a + num_b
    elif op == "subtract":
        result = num_a - num_b
    elif op == "multiply":
        result = num_a * num_b
    elif op == "divide":
        if num_b == 0:
            return "Error: Division by zero is not allowed."
        result = num_a / num_b
    else:
        return "Error: Unsupported operation. Use 'add', 'subtract', 'multiply', or 'divide'."

    return str(result)


@tool
def get_weather(city: str) -> str:
    """Return a simulated weather report for the given city."""
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
    """Return a simulated web search result for the given query."""
    # Simulated search result
    return f"Top result for '{query}': This is a simulated web search result."


# ==================== Agent Factory ====================

def create_agent(
    tools: List[BaseTool],
    llm: Runnable,
    system_prompt: Optional[str] = None,
    verbose: bool = True,
) -> AgentExecutor:
    """
    Create a ReAct agent executor with the given tools and language model.

    The tools are bound to the agent in two ways:
    1. `create_react_agent` receives them so their names and docstrings are
       included in the prompt, letting the LLM decide which tool to call.
    2. `AgentExecutor` receives the same tools so it can execute the selected
       action.

    Args:
        tools (List[BaseTool]): List of tools available to the agent.
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

    # Create the agent. Passing tools here binds their names and descriptions
    # into the prompt so the model can choose the right tool.
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
        output_parser=ReActSingleInputOutputParser(),
    )

    # Create and return the executor. The executor also receives the tools so
    # it can actually run the tool selected by the agent.
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
    Errors raised during invocation are caught and returned as a message
    so the caller can handle them gracefully.

    Args:
        agent_executor (AgentExecutor): The configured agent executor.
        query (str): The user's question or instruction.

    Returns:
        str: The agent's final answer, or an error message if invocation fails.
    """
    print(f"\n--- Query: {query} ---")
    try:
        response = agent_executor.invoke({"input": query})
        answer = response.get("output", "No output produced.")
    except Exception as exc:
        print(f"Error invoking agent: {exc}")
        answer = f"An error occurred while processing the query: {exc}"
    print(f"Answer: {answer}")
    return answer


# ==================== Main Execution ====================

if __name__ == "__main__":
    # Define the tools list with expanded descriptions
    # To add a new tool:
    # 1. Define a function with @tool decorator, type hints, and a concise
    #    docstring that describes what it does and the arguments it expects.
    # 2. Add the function to the `tools` list below.
    tools: List[BaseTool] = [
        calculator,
        get_weather,
        search_web,
    ]

    # Initialize the language model
    llm = ChatOpenAI(model="gpt-4", temperature=0)

    # Create the agent using the factory function. The tools are bound to the
    # agent here: create_react_agent exposes them to the model, and the
    # AgentExecutor uses them to run the chosen actions.
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
