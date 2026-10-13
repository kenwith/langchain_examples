"""Example 69: Ollama Async Structured Output with Validation

| Model | Provider | Description |
|-------|----------|-------------|
| llama3.2 | Ollama | Async structured output with Pydantic validation |

This example demonstrates how to combine async execution, Ollama,
structured output, and validation using `init_chat_model` and Pydantic.
"""

import asyncio
import os

from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field, field_validator


class MovieReview(BaseModel):
    """Structured output for a movie review."""

    title: str = Field(description="The movie title")
    rating: float = Field(description="Rating from 0 to 10")
    summary: str = Field(description="One-sentence summary")
    genres: list[str] = Field(description="List of genres")

    @field_validator("rating")
    @classmethod
    def validate_rating(cls, value: float) -> float:
        """Ensure the rating is within the accepted range."""
        if value < 0 or value > 10:
            raise ValueError("Rating must be between 0 and 10")
        return value


async def generate_movie_review(movie: str) -> MovieReview:
    """Generate a structured movie review using Ollama asynchronously."""
    model = init_chat_model(
        model=os.getenv("OLLAMA_MODEL", "llama3.2"),
        model_provider="ollama",
        temperature=0,
    )
    structured_model = model.with_structured_output(MovieReview)
    response = await structured_model.ainvoke(
        f"Review the movie '{movie}'. Return a structured result."
    )
    return response


async def main() -> None:
    """Run the example and print the validated structured output."""
    try:
        review = await generate_movie_review("Inception")
        print(review.model_dump())
    except Exception as exc:
        print(f"Validation or execution error: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
