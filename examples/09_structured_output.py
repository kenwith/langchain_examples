"""Example of structured output with Pydantic.

This module demonstrates how to define a Pydantic model with additional fields,
document the schema, and validate input data.
"""

from pydantic import BaseModel, Field, ValidationError, field_validator


class Person(BaseModel):
    """A person with basic information.

    Attributes:
        name: Full name of the person.
        age: Age in years.
        email: Email address.
        tags: List of tags associated with the person.
    """

    name: str = Field(..., description="Full name of the person")
    age: int = Field(..., gt=0, description="Age in years")
    email: str = Field(..., description="Email address")
    tags: list[str] = Field(default_factory=list, description="List of tags")

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        """Validate that the email contains an @ symbol."""
        if "@" not in v:
            raise ValueError("Email must contain @")
        return v


def main() -> None:
    """Demonstrate schema definition and validation."""
    # Valid input
    person = Person(name="Alice", age=30, email="alice@example.com", tags=["admin"])
    print(person)

    # Invalid input: negative age
    try:
        Person(name="Bob", age=-5, email="bob@example.com")
    except ValidationError as e:
        print("Validation error:", e)


if __name__ == "__main__":
    main()
