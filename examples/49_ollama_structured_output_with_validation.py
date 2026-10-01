# | Example | Description |
# |---------|-------------|
# | 49 | Ollama Structured Output with Validation |
"""
Example 49: Ollama Structured Output with Validation
====================================================
This example demonstrates how to get validated structured output from an
Ollama chat model using a Pydantic schema. The schema includes field
validators to ensure the model's response conforms to expected constraints.
"""

import os

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, field_validator


class MovieReview(BaseModel):
    """Schema for a movie review with validation constraints."""

    title: str = Field(description="The title of the movie")
    year: int = Field(description="The release year of the movie")
    rating: float = Field(description="Rating from 1.0 to 10.0")
    summary: str = Field(description="A brief summary of the review")

    @field_validator("year")
    @classmethod
    def validate_year(cls, v: int) -> int:
        if v < 1888 or v > 2025:
            raise ValueError("Year must be between 1888 and 2025")
        return v

    @field_validator("rating")
    @classmethod
    def validate_rating(cls, v: float) -> float:
        if v < 1.0 or v > 10.0:
            raise ValueError("Rating must be between 1.0 and 10.0")
        return v


def run_example() -> None:
    """Run the structured output example with an Ollama chat model."""
    # Use environment variables for configuration, with sensible local defaults.
    model_name = os.getenv("OLLAMA_MODEL", "llama3.1")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Initialize the chat model provider-agnostically.
    model = init_chat_model(
        model_name,
        model_provider="ollama",
        base_url=base_url,
    )

    # Attach the Pydantic schema to the model for structured output.
    structured_llm = model.with_structured_output(MovieReview)

    # Create a prompt template that asks for a movie review.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a movie critic. Always respond in JSON format matching the "
                "provided schema.",
            ),
            ("human", "Review the movie {movie}."),
        ]
    )

    # Chain the prompt and the structured model.
    chain = prompt | structured_llm

    # Invoke the chain with a movie title.
    result = chain.invoke({"movie": "Inception"})

    # The result is already a validated MovieReview instance.
    print("Structured output (validated):")
    print(f"  Title:   {result.title}")
    print(f"  Year:    {result.year}")
    print(f"  Rating:  {result.rating}")
    print(f"  Summary: {result.summary}")


if __name__ == "__main__":
    run_example()
