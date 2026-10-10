import math
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.llms import OpenAI


def calculator(expression: str) -> str:
    """Safely evaluate a mathematical expression."""
    try:
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def main():
    tools = [
        Tool(
            name="Calculator",
            func=calculator,
            description="Useful for math calculations. Input should be a mathematical expression.",
        )
    ]

    llm = OpenAI(temperature=0)
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )

    question = "What is 2 + 2?"
    answer = agent.run(question)
    print(answer)


if __name__ == "__main__":
    main()
