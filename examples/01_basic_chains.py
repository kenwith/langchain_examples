"""Basic chain example with a reusable response formatter.

This example demonstrates how to create and run a simple LangChain chain.
It also defines a ``print_response`` function that can be reused to format
model output consistently.

Usage:
    Run the script directly:

    ```bash
    python examples/01_basic_chains.py
    ```

    The script will print the model's response with a simple visual separator.

Example:
    ```python
    from langchain.llms import OpenAI
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate

    llm = OpenAI(temperature=0)
    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run("eco-friendly water bottles")
    print_response(response)
    ```
"""

from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


def print_response(response: str) -> None:
    """Print a formatted response to the console.

    Args:
        response: The model output to print.
    """
    print("\n" + "=" * 60)
    print("Response:")
    print("=" * 60)
    print(response)
    print("=" * 60 + "\n")


def main() -> None:
    """Run a basic chain example."""
    llm = OpenAI(temperature=0)
    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run("eco-friendly water bottles")
    print_response(response)


if __name__ == "__main__":
    main()
