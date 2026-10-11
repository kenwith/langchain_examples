"""Example 67: Async Structured Output with Ollama

| Field         | Value                                    |
|---------------|------------------------------------------|
| Model         | llama3 (or OLLAMA_MODEL env var)         |
| Provider      | Ollama                                   |
| Output Schema | Person (Pydantic model)                  |

This example demonstrates how to use Ollama with LangChain's
`init_chat_model` to perform structured output extraction asynchronously.
A Pydantic schema defines the expected output structure.
"""

import asyncio
import os

from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model


class Person(BaseModel):
    """Schema for extracted person information."""

    name: str = Field(description="The person's full name")
    age: int = Field(description="The person's age in years")


async def extract_person(text: str) -> Person:
    """Extract a Person object from the given text using Ollama."""
    model = init_chat_model(
        model=os.getenv("OLLAMA_MODEL", "llama3"),
        provider="ollama",
    )
    structured_model = model.with_structured_output(Person)
    return await structured_model.ainvoke(text)


async def main() -> None:
    """Run the async structured output example."""
    sample_text = "John Doe is a 30-year-old software engineer."
    person = await extract_person(sample_text)
    print(person)


if __name__ == "__main__":
    asyncio.run(main())
