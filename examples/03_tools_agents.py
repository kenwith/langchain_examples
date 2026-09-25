"""
Custom tools for LangChain agents.

This example demonstrates how to define and register custom tools with
an OpenAI-powered agent. It shows two styles:

1. Subclassing BaseTool (CalculatorTool, StringLengthTool) for tools that
   need explicit Pydantic schemas or custom validation.
2. Using the `@tool` decorator (reverse_string) to turn a plain Python
   function into a tool, letting LangChain infer the schema from the
   function signature.

Each tool declares:
- name: a unique identifier the agent can reference.
- description: a natural-language description the LLM uses to decide when to use it.
- args_schema: a Pydantic model describing the expected input (for BaseTool subclasses).
- _run: the synchronous implementation (for BaseTool subclasses).
- _arun: the asynchronous implementation (for BaseTool subclasses).

Tool registration is handled by passing tool instances to `initialize_agent`.
The agent will then be able to invoke them based on their descriptions.
Tool inputs are validated against their Pydantic args_schema before the tool's
_run method is called. Each tool is designed to gracefully handle errors by
catching exceptions and returning a meaningful message, so the agent can
recover and try alternative approaches.

Error handling around the entire agent execution is demonstrated in `main()`
with a try-except block that provides a fallback message.
"""
import os
import ast
import operator
import logging
from typing import Optional, Type, List, Dict

from langchain.tools import BaseTool, tool
from langchain.agents import initialize_agent, AgentType
from langchain.llms import OpenAI
from pydantic import BaseModel, Field

# Configure logging for better visibility of errors
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    expression: str = Field(
        description="The math expression to evaluate, e.g. '2 + 3 * 4'",
        min_length=1,
        max_length=200,
    )


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
        except SyntaxError as e:
            logger.error(f"Syntax error in calculator expression '{expression}': {e}")
            return (
                f"Invalid math expression: '{expression}'. "
                "Please use a valid mathematical expression with numbers and operators "
                "like +, -, *, /, **, and parentheses."
            )
        except (ValueError, ZeroDivisionError, TypeError) as e:
            logger.error(f"Evaluation error in calculator expression '{expression}': {e}")
            return (
                f"Error evaluating expression '{expression}': {str(e)}. "
                "Please check your math expression."
            )
        except Exception as e:
            logger.error(f"Unexpected error in calculator expression '{expression}': {e}")
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


class StringLengthInput(BaseModel):
    text: str = Field(
        description="The string to measure, e.g. 'hello'",
        max_length=100000,
    )


class StringLengthTool(BaseTool):
    name = "string_length"
    description = "Useful for when you need to know the number of characters in a string. Input should be a string."
    args_schema: Type[BaseModel] = StringLengthInput

    def _run(self, text: str) -> str:
        """Return the length of the input string, handling any unexpected errors."""
        try:
            return f"The length of the string is {len(text)} characters."
        except Exception as e:
            logger.error(f"Error in string_length tool: {e}")
            return f"Error computing string length: {str(e)}. Please provide a valid string."

    async def _arun(self, text: str) -> str:
        """Async version of _run."""
        return self._run(text)


@tool
def reverse_string(text: str) -> str:
    """Reverses the given string. Input should be a string."""
    try:
        return text[::-1]
    except Exception as e:
        logger.error(f"Error in reverse_string tool: {e}")
        return f"Error reversing string: {str(e)}. Please provide a valid string."


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
    """
    Run an agent with custom calculator, string-length, and reverse-string tools.

    Tool registration is handled by passing tool instances to `initialize_agent`.
    The calculator and string-length tools are BaseTool subclasses with explicit
    Pydantic schemas. The reverse-string tool is a plain function decorated with
    `@tool`, demonstrating how agents interact with user-defined functions.
    The LLM reads the tool descriptions to decide when to invoke each tool.

    Each tool's `_run` method contains try/except blocks to gracefully handle
    errors and return meaningful messages, ensuring the agent can continue
    processing even if a tool fails.

    Execution of the agent is wrapped in a try-except block to catch any
    unexpected errors (e.g., LLM parsing failures) and provide a fallback message.
    """
    # Load API key from environment - never hardcode credentials
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")

    llm = OpenAI(api_key=api_key, temperature=0)
    tools = [CalculatorTool(), StringLengthTool(), reverse_string]

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        return_intermediate_steps=True,  # capture tool calls for inspection
    )

    try:
        # Example valid queries
        print(agent.run("What is 12 * 8 + 4?"))
        print(agent.run("Calculate (3 + 5) ** 2"))
        print(agent.run("What is the length of the word 'hello'?"))
        print(agent.run("Reverse the string 'hello'"))

        # Example invalid expression to demonstrate helpful error
        print(agent.run("What is 2 +* 3?"))
    except Exception as e:
        logger.error(f"Agent execution error: {e}")
        print(f"An error occurred while running the agent: {e}")
        print("Fallback message: Please check the input and try again.")

    # Show how to extract tool call arguments from the agent's response
    # (using __call__ to get intermediate steps)
    response = agent({"input": "What is 12 * 8 + 4?"})
    calls = extract_tool_call_arguments(response)
    print("\nExtracted tool calls from agent response:")
    for call in calls:
        print(f"  Tool: {call['tool']}, Args: {call['tool_input']}")


if __name__ == "__main__":
    main()
