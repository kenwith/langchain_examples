"""
| # | Example | Description |
|---|---------|-------------|
| 42 | Ollama Structured Output | Use init_chat_model with Ollama and Pydantic schema |

This example demonstrates how to use `init_chat_model` with `model_provider="ollama"` and `with_structured_output()` to get structured responses as Pydantic objects.
"""

import os

from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field


class Joke(BaseModel):
    """Schema for a joke."""

    setup: str = Field(..., description="The setup of the joke")
    punchline: str = Field(..., description="The punchline of the joke")
    rating: int = Field(..., description="A rating from 1 to 10")


def main() -> None:
    """Run the Ollama structured output example."""
    model = init_chat_model(
        model=os.getenv("OLLAMA_MODEL", "llama3.1"),
        model_provider="ollama",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0.0,
    )

    structured_model = model.with_structured_output(Joke)
    result = structured_model.invoke("Tell me a funny joke")
    print(result)


if __name__ == "__main__":
    main()
