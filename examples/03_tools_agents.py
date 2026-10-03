"""
Custom tools for LangChain agents.

This example demonstrates how to define and register custom tools with an
OpenAI-powered agent. It shows two idiomatic styles:

1. Subclassing `BaseTool` (CalculatorTool, StringLengthTool) for tools that
   need explicit Pydantic schemas, custom validation, or async support.
2. Using the `@tool` decorator (reverse_string, word_count) to turn plain
   Python functions into tools, letting LangChain infer the schema from the
   function signature and docstring.

Each tool declares:
- name: a unique identifier the agent can reference.
- description: a natural-language description the LLM uses to decide when to use it.
- args_schema: a Pydantic model describing the expected input (for BaseTool subclasses).
- _run: the synchronous implementation (for BaseTool subclasses).
- _arun: the asynchronous implementation (for BaseTool subclasses).

Tool registration is handled by passing tool instances (or @tool-decorated
functions) to `initialize_agent`. The agent's LLM reads each tool's name and
description to decide when to invoke it. Tool inputs are validated against
their Pydantic args_schema before the tool's _run method is called. Each tool
is designed to gracefully handle errors by catching exceptions and returning a
meaningful message, so the agent can recover and try alternative approaches.

The example also demonstrates how to inspect the intermediate steps of an agent
execution by setting `return_intermediate_steps=True` and using a helper function
(`extract_tool_call_arguments`) to parse the tool calls from the response.

Requirements:
- Set the OPENAI_API_KEY environment variable to your OpenAI API key.
- Install the required dependencies: `pip install langchain openai pydantic`.

Usage:
    python examples/03_tools_agents.py

Error handling around tool execution is demonstrated in `run_agent()`, a helper
that wraps each call to the agent, catches any exceptions, and returns a readable
error message so the user always gets a clear response.

Security note:
- Never hardcode API keys or other secrets. Always load credentials from
  environment variables or a secure secret manager.
"""
import os
import ast
import operator
import logging
from typing import Any, Callable, Dict, List, Type, Union

from langchain.tools import BaseTool, tool
from langchain.agents import initialize_agent, AgentType
from langchain.llms import OpenAI
from pydantic import BaseModel, Field

# Configure logging for better visibility of errors
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Allowed AST operators for safe evaluation
ALLOWED_OPERATORS: Dict[type, Callable[..., Any]] = {
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
    """Pydantic schema for the CalculatorTool input."""

    expression: str = Field(
        description="The math expression to evaluate, e.g. '2 + 3 * 4'",
        min_length=1,
        max_length=200,
    )


class CalculatorTool(BaseTool):
    """Tool for safely evaluating mathematical expressions.

    This tool uses Python's AST module to parse and evaluate expressions
    with a restricted set of operators. It is designed to handle invalid
    input gracefully and return a readable error message.
    """

    name: str = "calculator"
    description: str = (
        "Useful for when you need to answer questions about math. "
        "Input should be a valid mathematical expression."
    )
    args_schema: Type[BaseModel] = CalculatorInput

    def _run(self, expression: str) -> str:
        """Evaluate a math expression safely and return a helpful error on invalid input.

        Args:
            expression: A string containing a mathematical expression.

        Returns:
            A string describing the result, or an error message if evaluation fails.
        """
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
        """Async version of _run.

        Args:
            expression: A string containing a mathematical expression.

        Returns:
            A string describing the result, or an error message if evaluation fails.
        """
        return self._run(expression)

    def _eval_node(self, node: ast.AST) -> Union[int, float]:
        """Recursively evaluate an AST node using only allowed operators.

        Args:
            node: The AST node to evaluate.

        Returns:
            The numeric result of evaluating the node.

        Raises:
            ValueError: If the node contains an unsupported operation or constant.
        """
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
    """Pydantic schema for the StringLengthTool input."""

    text: str = Field(
        description="The string to measure, e.g. 'hello'",
        max_length=100000,
    )


class StringLengthTool(BaseTool):
    """Tool for counting characters in a string.

    This tool returns the number of characters in the input text, including
    spaces and punctuation. It handles invalid inputs gracefully.
    """

    name: str = "string_length"
    description: str = (
        "Useful for when you need to know the number of characters in a string. "
        "Input should be a string."
    )
    args_schema: Type[BaseModel] = StringLengthInput

    def _run(self, text: str) -> str:
        """Return the length of the input string, handling any unexpected errors.

        Args:
            text: The string to measure.

        Returns:
            A string describing the character count, or an error message if the
            input is invalid.
        """
        try:
            return f"The length of the string is {len(text)} characters."
        except Exception as e:
            logger.error(f"Error in string_length tool: {e}")
            return f"Error computing string length: {str(e)}. Please provide a valid string."

    async def _arun(self, text: str) -> str:
        """Async version of _run.

        Args:
            text: The string to measure.

        Returns:
            A string describing the character count, or an error message if the
            input is invalid.
        """
        return self._run(text)


# Using the @tool decorator to turn plain Python functions into LangChain tools.
# LangChain infers the schema from the function signature and docstring.
# These decorated functions are bound to the agent exactly like BaseTool
# instances when included in the tools list passed to initialize_agent.
@tool
def reverse_string(text: str) -> str:
    """Reverses the given string.

    This tool is useful when the agent needs to reverse the characters in a text value.

    Args:
        text: The string to reverse. Must be a plain string.

    Returns:
        The reversed string. If the input is not a string or an unexpected error
        occurs, returns a descriptive error message so the agent can recover.
    """
    if not isinstance(text, str):
        logger.error(f"reverse_string received non-string input: {type(text).__name__}")
        return (
            f"Error reversing string: expected a string but got {type(text).__name__}. "
            "Please provide a valid string."
        )

    try:
        return text[::-1]
    except Exception as e:
        logger.error(f"Error in reverse_string tool: {e}")
        return f"Error reversing string: {str(e)}. Please provide a valid string."


@tool
def word_count(text: str) -> str:
    """Counts the number of words in the given text.

    This tool is useful when the agent needs to know how many words are in a
    sentence or paragraph. Words are separated by whitespace.

    Args:
        text: The text to count words in. Must be a plain string.

    Returns:
        A string describing the word count. If the input is not a string or an
        unexpected error occurs, returns a descriptive error message so the
        agent can recover.
    """
    if not isinstance(text, str):
        logger.error(f"word_count received non-string input: {type(text).__name__}")
        return (
            f"Error counting words: expected a string but got {type(text).__name__}. "
            "Please provide a valid string."
        )

    try:
        words = text.split()
        return f"The text contains {len(words)} words."
    except Exception as e:
        logger.error(f"Error in word_count tool: {e}")
        return f"Error counting words: {str(e)}. Please provide a valid string."


def extract_tool_call_arguments(agent_response: Dict[str, Any]) -> List[Dict[str, str]]:
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

    tool_calls = []
    for agent_action, _ in agent_response["intermediate_steps"]:
        tool_calls.append({"tool": agent_action.tool, "tool_input": agent_action.tool_input})
    return tool_calls


def run_agent(agent: Any, query: str) -> str:
    """
    Run a single query through the agent and return a readable response.

    This helper wraps tool execution so that any exception raised while the
    agent is running (including tool errors, parsing failures, or API issues)
    is caught and converted into a readable error message. This ensures the
    caller always receives a string response instead of an unhandled exception.

    Args:
        agent: The initialized LangChain agent.
        query: The user query to send to the agent.

    Returns:
        The agent's response as a string, or a readable error message if the
        agent execution fails.
    """
    try:
        return agent.run(query)
    except Exception as e:
        logger.error(f"Agent execution error for query '{query}': {e}")
        return (
            f"An error occurred while processing your request: {type(e).__name__}: {e}. "
            "Please check the input and try again."
        )


def main() -> None:
    """
    Run an agent with custom calculator, string-length, reverse-string, and word-count tools.

    Tool registration is handled by passing tool instances to `initialize_agent`.
    The calculator and string-length tools are BaseTool subclasses with explicit
    Pydantic schemas. The reverse-string and word-count tools are plain functions
    decorated with `@tool`, demonstrating how agents interact with user-defined
    functions. The LLM reads the tool descriptions to decide when to invoke each tool.

    Each tool's `_run` method contains try/except blocks to gracefully handle
    errors and return meaningful messages, ensuring the agent can continue
    processing even if a tool fails.

    Each agent call is wrapped by the `run_agent()` helper, which catches any
    unexpected exceptions and returns a readable error message instead of
    crashing the script.
    """
    # Load API key from environment - never hardcode credentials
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")

    llm = OpenAI(api_key=api_key, temperature=0)

    # Bind the tools to the agent by passing them as the first argument to
    # initialize_agent. The agent's LLM reads each tool's name and description
    # to decide when to invoke it. The tool list can contain a mix of BaseTool
    # instances and @tool-decorated functions; LangChain normalizes them into
    # the same internal representation.
    tools = [CalculatorTool(), StringLengthTool(), reverse_string, word_count]

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        return_intermediate_steps=True,  # capture tool calls for inspection
    )

    sample_query = "What is 12 * 8 + 4?"

    # Example valid queries
    print(f"Sample query: {sample_query}")
    print(run_agent(agent, sample_query))
    print(run_agent(agent, "Calculate (3 + 5) ** 2"))
    print(run_agent(agent, "What is the length of the word 'hello'?"))
    print(run_agent(agent, "Reverse the string 'hello'"))
    print(run_agent(agent, "How many words are in the sentence 'The quick brown fox jumps over the lazy dog'?"))

    # Example invalid expression to demonstrate helpful error handling
    print(run_agent(agent, "What is 2 +* 3?"))

    # Show how to extract tool call arguments from the agent's response
    # (using __call__ to get intermediate steps)
    agent_response = agent({"input": sample_query})
    tool_calls = extract_tool_call_arguments(agent_response)
    print("\nExtracted tool calls from agent response:")
    for tool_call in tool_calls:
        print(f"  Tool: {tool_call['tool']}, Args: {tool_call['tool_input']}")


if __name__ == "__main__":
    main()
