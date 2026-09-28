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

    name: str = Field(..., min_length=1, description="Full name of the person")
    age: int = Field(..., gt=0, le=150, description="Age in years")
    email: str = Field(..., description="Email address")
    tags: list[str] = Field(default_factory=list, description="List of tags")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate that the name is not empty or whitespace."""
        v = v.strip()
        if not v:
            raise ValueError("Name must not be empty")
        return v

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        """Validate that the email contains an @ symbol and normalize it."""
        v = v.strip().lower()
        if "@" not in v:
            raise ValueError("Email must contain @")
        return v

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: list[str]) -> list[str]:
        """Validate that tags are non-empty and stripped."""
        cleaned = []
        for tag in v:
            tag = tag.strip()
            if not tag:
                raise ValueError("Tags must not be empty")
            cleaned.append(tag)
        return cleaned


def main() -> None:
    """Demonstrate schema definition and validation."""
    # Valid input
    person = Person(
        name="Alice",
        age=30,
        email="alice@example.com",
        tags=["admin", "user"],
    )
    print(person)

    # Invalid input: negative age
    try:
        Person(name="Bob", age=-5, email="bob@example.com")
    except ValidationError as e:
        print("Validation error:", e)

    # Invalid input: empty name
    try:
        Person(name="   ", age=30, email="bob@example.com")
    except ValidationError as e:
        print("Validation error:", e)

    # Invalid input: invalid email
    try:
        Person(name="Bob", age=30, email="invalid-email")
    except ValidationError as e:
        print("Validation error:", e)

    # Invalid input: empty tag
    try:
        Person(name="Bob", age=30, email="bob@example.com", tags=["admin", ""])
    except ValidationError as e:
        print("Validation error:", e)


if __name__ == "__main__":
    main()
