"""Example: Async Custom Output Parser
=====================================

This example demonstrates how to create a custom asynchronous output parser
for handling structured responses from a chat model. The parser expects the
model to return a JSON object with fields `name` and `age`, and converts it
into a dictionary. The example uses the provider-agnostic `init_chat_model`
function, so it works with any supported chat model provider.

## Overview

- Defines a custom `AsyncJSONParser` that inherits from `BaseOutputParser`.
- Overrides the `parse` method to be async and handle JSON parsing.
- Provides `get_format_instructions` to guide the model's output format.
- Uses `init_chat_model` to create a chat model without hardcoding a provider.
- Runs an asynchronous main function that sends a prompt and parses the response.

## Setup

- Install the required packages: `langchain`, `langchain-core`, and a model provider package (e.g., `openai`).
- Set the appropriate environment variable for the model API key (e.g., `OPENAI_API_KEY`).

## Usage

Run the script directly. The example will ask the model to generate information
about a person and then parse the structured response.

```bash
python examples/100_async_custom_output_parser.py
```
"""

import asyncio
import os
from typing import Any, Dict

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import BaseOutputParser


class AsyncJSONParser(BaseOutputParser[Dict[str, Any]]):
    """Custom asynchronous output parser that expects a JSON object with 'name' and 'age'."""

    async def parse(self, text: str) -> Dict[str, Any]:
        """Parse the model output asynchronously."""
        # In a real implementation, this would use a JSON parser with error handling.
        # For simplicity, we assume the output is valid JSON.
        import json

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON output: {e}") from e

        if "name" not in data or "age" not in data:
            raise ValueError("Output must contain 'name' and 'age' fields.")

        return data

    def get_format_instructions(self) -> str:
        """Return instructions for the model to produce the expected format."""
        return (
            "Return a JSON object with the following fields:\n"
            '{"name": string, "age": integer}\n'
            "Do not include any other text."
        )


async def run_example() -> None:
    """Run the async custom output parser example."""
    # Initialize the chat model (provider-agnostic).
    # The model name determines the provider; ensure the corresponding API key is set.
    model = init_chat_model(model="gpt-4o-mini", temperature=0)

    # Create a prompt that asks for structured information.
    prompt = (
        "Provide the name and age of a famous scientist. "
        "Follow the format instructions:\n"
        + AsyncJSONParser().get_format_instructions()
    )

    # Invoke the model asynchronously.
    response = await model.ainvoke([HumanMessage(content=prompt)])

    # Extract the text content from the response.
    text = response.content

    # Parse the output using the custom async parser.
    parser = AsyncJSONParser()
    parsed_data = await parser.parse(text)

    # Print the parsed result.
    print("Parsed response:", parsed_data)


def main() -> None:
    """Entry point for the example."""
    asyncio.run(run_example())


if __name__ == "__main__":
    main()
