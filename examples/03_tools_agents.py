import os
import ast
import operator
from typing import Optional, Type, List, Dict

from langchain.tools import BaseTool
from langchain.agents import initialize_agent, AgentType
from langchain.llms import OpenAI
from pydantic import BaseModel, Field

# Allowed AST operators for safe evaluation
ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}


class CalculatorInput(BaseModel):
    expression: str = Field(description="The math expression to evaluate, e.g. '2 + 3 * 4'")


class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Useful for when you need to answer questions about math. Input should be a valid mathematical expression."
    args_schema: Type[BaseModel] = CalculatorInput

    def _run(self, expression: str) -> str:
        """Evaluate a math expression safely and return a helpful error on invalid input."""
        try:
            # Parse the expression into an AST to validate syntax and allowed operations
            tree = ast.parse(expression, mode="eval")
            result = self._eval_node(tree.body)
            return f"The result is {result}"
        except SyntaxError:
            return (
                f"Invalid math expression: '{expression}'. "
                "Please use a valid mathematical expression with numbers and operators "
                "like +, -, *, /, **, and parentheses."
            )
        except (ValueError, ZeroDivisionError, TypeError) as e:
            return (
                f"Error evaluating expression '{expression}': {str(e)}. "
                "Please check your math expression."
            )
        except Exception as e:
            return (
                f"Unexpected error evaluating expression '{expression}': {str(e)}. "
                "Please try a simpler expression."
            )

    async def _arun(self, expression: str) -> str:
        """Async version of _run."""
        return self._run(expression)

    def _eval_node(self, node):
        """Recursively evaluate an AST node using only allowed operators."""
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant: {node.value}")

        if isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op_type = type(node.op)
            if op_type not in ALLOWED_OPERATORS:
                raise ValueError(f"Unsupported operator: {op_type.__name__}")
            return ALLOWED_OPERATORS[op_type](left, right)

        if isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            op_type = type(node.op)
            if op_type not in ALLOWED_OPERATORS:
                raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
            return ALLOWED_OPERATORS[op_type](operand)

        raise ValueError(f"Unsupported expression element: {type(node).__name__}")


def extract_tool_call_arguments(agent_response: dict) -> List[Dict[str, str]]:
    """
    Extract tool call arguments from an agent response dict.

    This helper expects the response returned by an AgentExecutor when
    `return_intermediate_steps=True`. The response should contain an
    'intermediate_steps' key with a list of (AgentAction, observation) tuples.

    Args:
        agent_response: The full response dict from `agent({"input": ...})`.

    Returns:
        A list of dictionaries, each with 'tool' and 'tool_input' keys.

    Raises:
        ValueError: If the response does not contain intermediate_steps.
    """
    if not isinstance(agent_response, dict) or "intermediate_steps" not in agent_response:
        raise ValueError(
            "Expected agent response dict with 'intermediate_steps' key. "
            "Make sure to set return_intermediate_steps=True on the agent."
        )

    calls = []
    for action, _observation in agent_response["intermediate_steps"]:
        calls.append({"tool": action.tool, "tool_input": action.tool_input})
    return calls


def main():
    """Run a simple agent with the calculator tool."""
    # Load API key from environment - never hardcode credentials
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")

    llm = OpenAI(api_key=api_key, temperature=0)
    tools = [CalculatorTool()]

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        return_intermediate_steps=True,  # capture tool calls for inspection
    )

    # Example valid queries
    print(agent.run("What is 12 * 8 + 4?"))
    print(agent.run("Calculate (3 + 5) ** 2"))

    # Example invalid expression to demonstrate helpful error
    print(agent.run("What is 2 +* 3?"))

    # Show how to extract tool call arguments from the agent's response
    # (using __call__ to get intermediate steps)
    response = agent({"input": "What is 12 * 8 + 4?"})
    calls = extract_tool_call_arguments(response)
    print("\nExtracted tool calls from agent response:")
    for call in calls:
        print(f"  Tool: {call['tool']}, Args: {call['tool_input']}")


if __name__ == "__main__":
    main()
