"""
Example: Custom Output Parser for Comma-Separated Lists
"""

from langchain_core.output_parsers import BaseOutputParser
from langchain_chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate


class CommaSeparatedListOutputParser(BaseOutputParser[list[str]]):
    """
    Custom output parser that converts a comma-separated string into a list.
    """

    def parse(self, text: str) -> list[str]:
        """
        Parse the text into a list of items separated by commas.
        """
        # Split on commas, strip whitespace, and filter out empty strings
        items = [item.strip() for item in text.split(",")]
        return [item for item in items if item]


def run_example() -> None:
    """
    Demonstrate the custom output parser with a chat model.
    """
    # Initialize a chat model (provider-agnostic)
    model = init_chat_model(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),  # Placeholder only
    )

    # Create a prompt that asks for a comma-separated list
    prompt = PromptTemplate.from_template(
        "List three fruits as a comma-separated list (no extra text):"
    )

    # Chain the prompt, model, and parser
    chain = prompt | model | CommaSeparatedListOutputParser()

    # Invoke the chain
    result = chain.invoke({})
    print("Parsed list:", result)


if __name__ == "__main__":
    run_example()
