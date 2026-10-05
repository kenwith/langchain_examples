"""Module docstring for structured output example."""
from typing import Literal
from pydantic import BaseModel, Field, ValidationError
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


class Joke(BaseModel):
    """Joke structure."""
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline of the joke")
    rating: Literal["funny", "not funny"] = Field(
        description="How funny the joke is"
    )


def main() -> None:
    llm = ChatOpenAI(model="gpt-4o-mini")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a comedian. Output only JSON."),
            ("human", "Tell me a joke about {topic}"),
        ]
    )
    chain = prompt | llm.with_structured_output(Joke)
    try:
        result = chain.invoke({"topic": "programmers"})
        print(result)
    except ValidationError as e:
        print("Validation error:", e)


if __name__ == "__main__":
    main()
