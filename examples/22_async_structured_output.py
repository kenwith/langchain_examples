"""Example 22: Async Structured Output.

| Example | Description |
|---------|-------------|
| 22      | Combine asyncio with structured output using a Pydantic model. |
"""

import asyncio
import os
from typing import List

from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field


class Person(BaseModel):
    """A person's name and age extracted from text."""

    name: str = Field(description="The person's full name")
    age: int = Field(description="The person's age in years")


async def extract_person(text: str, model) -> Person:
    """Extract a Person from the given text using the chat model."""
    structured_model = model.with_structured_output(Person)
    response = await structured_model.ainvoke(text)
    return response


async def main() -> None:
    """Run concurrent structured extractions on sample texts."""
    # Use environment variables for credentials, never hardcode them.
    model = init_chat_model(
        model=os.getenv("MODEL", "gpt-4o-mini"),
        provider=os.getenv("MODEL_PROVIDER", "openai"),
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0,
    )

    texts = [
        "Alice Johnson is 30 years old.",
        "Bob Smith is 25 years old.",
        "Carol White is 42 years old.",
    ]

    # Process all texts concurrently.
    people: List[Person] = await asyncio.gather(
        *(extract_person(text, model) for text in texts)
    )

    for person in people:
        print(f"Extracted: {person.name}, age {person.age}")


if __name__ == "__main__":
    asyncio.run(main())
