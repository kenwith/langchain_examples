"""
Example: Structured Output with Validation
==========================================

This example demonstrates how to use `with_structured_output` with a Pydantic model
that includes field validators. We'll define a `Person` model with validation rules,
then ask a chat model to extract structured data from a text description.

The example is provider-agnostic: we use `init_chat_model` to create a chat model
based on environment variables (e.g., `OPENAI_API_KEY`, `MODEL_NAME`).
"""

from typing import List, Optional

from langchain.chat_models import init_chat_model
from pydantic import BaseModel, EmailStr, Field, field_validator


# -----------------------------------------------------------------------------
# Pydantic model with validators
# -----------------------------------------------------------------------------
class Person(BaseModel):
    """A person with validated fields."""

    name: str = Field(description="Full name of the person")
    age: int = Field(description="Age in years")
    email: EmailStr = Field(description="Email address")
    tags: List[str] = Field(default_factory=list, description="Optional tags")

    @field_validator("age")
    def age_must_be_positive(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Age must be a positive integer")
        return v

    @field_validator("name")
    def name_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Name cannot be empty")
        return v


# -----------------------------------------------------------------------------
# Main function demonstrating structured output with validation
# -----------------------------------------------------------------------------
def run_example() -> None:
    """Run the structured output example with validation."""
    # Initialize the chat model (provider and model are set via environment)
    model = init_chat_model()

    # Bind the structured output schema to the model
    structured_model = model.with_structured_output(Person)

    # Input text to extract from
    text = (
        "John Doe is a 30-year-old software engineer. "
        "His email is john.doe@example.com. "
        "He is interested in AI and Python."
    )

    # Invoke the model to get a validated Person instance
    person = structured_model.invoke(text)

    # Print the result
    print("Extracted person:")
    print(person)
    print("\nValidated fields:")
    print(f"Name: {person.name}")
    print(f"Age: {person.age}")
    print(f"Email: {person.email}")
    print(f"Tags: {person.tags}")


# -----------------------------------------------------------------------------
# Demo block
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    run_example()
