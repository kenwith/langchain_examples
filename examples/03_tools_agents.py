import math
from langchain.agents import initialize_agent, AgentType, AgentExecutor
from langchain.llms import OpenAI
from langchain.tools import tool


@tool("Calculator")
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    This tool is intended for arithmetic and mathematical calculations. The
    agent selects this tool when the question involves numeric expressions,
    formulas, or operations such as addition, subtraction, multiplication,
    division, powers, roots, or logarithms. The input is a string containing a
    mathematical expression, evaluated using Python's eval() with a restricted
    global namespace. The math module is available for functions such as sqrt,
    sin, cos, and log. If the expression is invalid or raises an exception, an
    error message is returned instead of raising an exception to the caller.

    Args:
        expression: A string representing a mathematical expression, e.g.,
            "2 + 2", "math.sqrt(16)", or "(3 * 4) / 2".

    Returns:
        A string representation of the result, or an error message if evaluation
        fails.
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers and return the product.

    This tool is registered as a LangChain tool and expects two integer inputs.
    The agent selects this tool when the task requires multiplying exactly two
    integer factors, such as calculating a product, total, area, or scaled
    quantity. The input should be a JSON string with keys "a" and "b".

    Args:
        a: The first integer factor.
        b: The second integer factor.

    Returns:
        The product of a and b as an integer.
    """
    return a * b


def _ensure_unique_tool_names(tools: list) -> list:
    """Ensure tool names are unique by appending a suffix if needed.

    This function checks for duplicate tool names and renames any conflicting
    tools by appending "_1", "_2", etc., until a unique name is found. The
    modification is performed in-place on the tool objects.

    Args:
        tools: A list of LangChain tool objects.

    Returns:
        The same list with unique tool names.
    """
    seen = set()
    for tool in tools:
        name = tool.name
        if name in seen:
            suffix = 1
            new_name = f"{name}_{suffix}"
            while new_name in seen:
                suffix += 1
                new_name = f"{name}_{suffix}"
            tool.name = new_name
        seen.add(tool.name)
    return tools


def build_agent() -> AgentExecutor:
    """Build and return a zero-shot ReAct agent with custom tools.

    This function constructs a LangChain agent using the
    ZERO_SHOT_REACT_DESCRIPTION agent type. It provides a custom prefix and
    suffix that define the prompt format, and registers two tools: a calculator
    for mathematical expressions and a multiplier for integer products. The
    tool names are checked for duplicates and renamed if necessary. The prefix
    is generated dynamically from the final tool names and descriptions.

    Returns:
        An initialized LangChain agent.
    """
    # The tool descriptions are the main signal the agent uses to decide which
    # tool to invoke for a given question. Each description should state the
    # purpose of the tool and the expected input format.
    tools = [
        calculator,
        multiply,
    ]

    # Ensure unique tool names to avoid conflicts.
    tools = _ensure_unique_tool_names(tools)

    # Build the prefix dynamically using the actual tool names and descriptions.
    tool_lines = "\n".join([f"{t.name}: {t.description}" for t in tools])
    tool_names = ", ".join([t.name for t in tools])
    prefix = f"""You are an AI assistant that uses tools to answer questions. You have access to the following tools:

{tool_lines}

Use the following format:
Question: the input question
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
"""

    suffix = """Begin!

Question: {input}
Thought:"""

    llm = OpenAI(temperature=0)
    # The ZERO_SHOT_REACT_DESCRIPTION agent determines which tool to use at
    # each step by reading the tool names and descriptions and selecting the
    # most relevant one for the observed question. It then formats the action
    # input according to the tool's expected schema.
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        prefix=prefix,
        suffix=suffix,
    )
    return agent


def main() -> None:
    """Build the agent and run it on a sample arithmetic question."""
    agent = build_agent()

    # The agent may iterate through several Thought/Action/Observation cycles
    # before it has enough information to produce a final answer.
    question = "What is 2 + 2?"
    answer = agent.run(question)
    print(answer)


if __name__ == "__main__":
    main()
