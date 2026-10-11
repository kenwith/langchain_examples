"""
Ollama Async Structured Output Test

This test verifies that the Ollama async structured output example returns a valid Pydantic object.

| Test | Description |
|------|-------------|
| test_ollama_async_structured_output | Verifies async structured output returns a valid Pydantic object |
"""

import asyncio
import os

from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field


class Joke(BaseModel):
    """A joke with a setup and punchline."""
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline of the joke")


def test_ollama_async_structured_output() -> None:
    """Test that the Ollama async structured output returns a valid Pydantic object."""
    result = asyncio.run(_run_async_structured_output())
    assert isinstance(result, Joke)
    assert result.setup
    assert result.punchline


async def _run_async_structured_output() -> Joke:
    """Run the async structured output example and return the result."""
    model = init_chat_model(
        model=os.getenv("OLLAMA_MODEL", "llama3.1"),
        model_provider="ollama",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    structured_model = model.with_structured_output(Joke)
    result = await structured_model.ainvoke(
        "Tell me a short joke. Return it as a JSON object with 'setup' and 'punchline'."
    )
    return result


if __name__ == "__main__":
    test_ollama_async_structured_output()
    print("Async structured output test passed.")
