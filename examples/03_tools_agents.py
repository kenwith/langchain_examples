"""Example 03: Tools and agents with graceful tool error handling.

This example introduces a self-contained calculator tool that can fail on
invalid input and shows how to catch exceptions inside the tool and return a
graceful message to the agent. It works fully offline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


@dataclass
class CalculationResult:
    """A calculation result returned by the calculator tool."""
    expression: str
    result: Optional[float] = None
    error: Optional[str] = None

    def __str__(self) -> str:
        if self.error:
            return f"Could not evaluate '{self.expression}': {self.error}"
        return f"{self.expression} = {self.result:g}"


@tool
def calculate(expression: str) -> CalculationResult:
    """Evaluate a mathematical expression.

    Args:
        expression: A mathematical expression string, e.g., '2 + 2' or 'sqrt(16)'.

    Returns:
        A CalculationResult with the evaluated result or an error message.
    """
    import math

    # Allow only safe builtins and useful math functions in the evaluation
    # namespace. No network access or dangerous builtins are available.
    allowed_globals = {
        "__builtins__": {},
        "math": math,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "pi": math.pi,
        "e": math.e,
    }

    try:
        # Evaluate the expression. The restricted globals prevent access to
        # dangerous functionality while still supporting arithmetic and math.
        result = eval(expression, allowed_globals, {})
        if isinstance(result, bool) or not isinstance(result, (int, float)):
            raise ValueError("Expression did not produce a number")
        return CalculationResult(expression=expression, result=float(result))
    except Exception as exc:
        # Catch the exception inside the tool and return a graceful message.
        return CalculationResult(expression=expression, error=str(exc))


def run_agent_without_executor() -> None:
    """Run a simple agent loop using a chat model and the calculator tool."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_with_tools = llm.bind_tools([calculate])

    messages = [
        HumanMessage(
            content="What is 1/0? Also, what is 123 * 456?"
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
                tool_result = calculate.invoke(tool_call["args"])
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
        [calculate],
        prompt,
    )
    executor = AgentExecutor(agent=agent, tools=[calculate], verbose=True)

    result = executor.invoke(
        {"input": "What is 1/0? Also, what is 123 * 456?"}
    )
    print("Agent output:", result["output"])


if __name__ == "__main__":
    # Uncomment one of the following:
    # run_agent_without_executor()
    run_agent_with_executor()
