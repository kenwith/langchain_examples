"""Example of structured output with a Pydantic model."""
from typing import Literal
from pydantic import BaseModel, Field, ValidationError, field_validator
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


class Joke(BaseModel):
    """Joke structure."""
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline of the joke")
    rating: Literal["funny", "not funny"] = Field(
        description="How funny the joke is"
    )

    @field_validator("setup", "punchline")
    @classmethod
    def check_non_empty(cls, v: str) -> str:
        """Ensure the field is not empty or whitespace-only."""
        if not v or not v.strip():
            raise ValueError("Field must not be empty")
        return v.strip()

    @field_validator("setup")
    @classmethod
    def check_setup_length(cls, v: str) -> str:
        """Ensure the setup has a reasonable length."""
        if len(v) < 3:
            raise ValueError("Setup must be at least 3 characters long")
        return v

    @field_validator("punchline")
    @classmethod
    def check_punchline_length(cls, v: str) -> str:
        """Ensure the punchline has a reasonable length."""
        if len(v) < 2:
            raise ValueError("Punchline must be at least 2 characters long")
        return v


def parse_joke_output(joke: Joke) -> dict:
    """Convert a Joke instance into a dictionary."""
    return joke.model_dump()


def main() -> None:
    llm = ChatOpenAI(model="gpt-4o-mini")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a comedian. Output only JSON."),
            ("human", "Tell me a joke about {topic}"),
        ]
    )
    # The Pydantic model defines the JSON schema and validates the LLM response.
    chain = prompt | llm.with_structured_output(Joke)
    try:
        result = chain.invoke({"topic": "programmers"})
        parsed_result = parse_joke_output(result)
        print(parsed_result)
    except ValidationError as e:
        print("Validation error:", e)


if __name__ == "__main__":
    main()
