import math
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.llms import OpenAI
from langchain.tools import tool


def calculator(expression: str) -> str:
    """Safely evaluate a mathematical expression."""
    try:
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers and return the product."""
    return a * b


def main():
    tools = [
        Tool(
            name="Calculator",
            func=calculator,
            description="Useful for math calculations. Input should be a mathematical expression.",
        ),
        multiply,
    ]

    prefix = """You are an AI assistant that uses tools to answer questions. You have access to the following tools:

Calculator: Useful for math calculations. Input should be a mathematical expression.
multiply: Multiplies two integers. Input should be a JSON string with keys "a" and "b", e.g., '{"a": 3, "b": 4}'.

Use the following format:
Question: the input question
Thought: you should always think about what to do
Action: the action to take, should be one of [Calculator, multiply]
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
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        prefix=prefix,
        suffix=suffix,
    )

    question = "What is 2 + 2?"
    answer = agent.run(question)
    print(answer)


if __name__ == "__main__":
    main()
