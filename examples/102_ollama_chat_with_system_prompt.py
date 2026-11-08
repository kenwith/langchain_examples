"""Ollama Chat with System Prompt.

| Provider | Model    | Description |
|----------|----------|-------------|
| Ollama   | llama3.2 | Use a system prompt with an Ollama chat model via `init_chat_model`. |

This example demonstrates how to use the provider-agnostic `init_chat_model`
factory to create an Ollama chat model and pass a system prompt through a
`ChatPromptTemplate`.
"""

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate


def create_chat_model():
    """Create an Ollama chat model."""
    return init_chat_model("llama3.2", model_provider="ollama")


def generate_rhyming_response(question: str) -> str:
    """Generate a rhyming response from the model using a system prompt."""
    model = create_chat_model()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that always answers in rhymes.",
            ),
            ("human", "{question}"),
        ]
    )
    chain = prompt | model
    return chain.invoke({"question": question}).content


def main() -> None:
    """Run a short demonstration of the Ollama chat model with a system prompt."""
    answer = generate_rhyming_response("What is the capital of France?")
    print(answer)


if __name__ == "__main__":
    main()
