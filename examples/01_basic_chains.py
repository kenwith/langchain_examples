"""Basic chain example with provider-agnostic setup using LCEL.

This module demonstrates the simplest usage of LangChain with the LangChain
Expression Language (LCEL). It builds a chain that takes a topic and returns a
fun fact about that topic.

The chain flow is as follows:
1. Initialize a language model using `init_chat_model`, which supports multiple
   providers (e.g., OpenAI, Anthropic) without changing the core logic. The
   provider and model name are read from environment variables, defaulting to
   OpenAI's GPT-3.5 Turbo.
2. Define a PromptTemplate that specifies the input variables and the template
   string.
3. Compose the prompt, the model, and an output parser into an LCEL chain using
   the pipe operator (`|`). LCEL makes it easy to combine components, add
   retries, fallbacks, and streaming, and to inspect the chain's steps.
4. Run the chain by calling `invoke` with a dictionary of input variables
   (e.g., `{"topic": "space"}`).
5. The chain formats the prompt, sends it to the LLM, parses the model output
   into a string, and returns the response.

Environment variables:
- MODEL_PROVIDER: (optional) The provider name, e.g., "openai" or "anthropic". Defaults to "openai".
- MODEL_NAME: (optional) The model identifier, e.g., "gpt-3.5-turbo". Defaults to "gpt-3.5-turbo".
- {PROVIDER}_API_KEY: The API key for the selected provider (e.g., OPENAI_API_KEY). Must be set.

Usage:
    Set the required API key for your chosen provider first. For example:

    export OPENAI_API_KEY="your-api-key"
    python examples/01_basic_chains.py

    Or use a different provider:

    export MODEL_PROVIDER="anthropic"
    export ANTHROPIC_API_KEY="your-api-key"
    export MODEL_NAME="claude-3-5-sonnet-20240620"
    python examples/01_basic_chains.py

    The script will print a fun fact about "space" by default.

This script provides the following functions:
- build_prompt(): returns a PromptTemplate for the chain.
- get_response(topic): builds and runs the LCEL chain, returning the response as a string.
- run_chain(topic): runs the chain for a given topic and prints the response.
- run_example(): runs a sample topic and prints a clear, labeled output.
- main(): entry point that calls run_example().
"""

import os
from langchain.chat_models import init_chat_model
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


def build_prompt() -> PromptTemplate:
    """Build the prompt template for the chain."""
    return PromptTemplate(
        input_variables=["topic"],
        template="Tell me a fun fact about {topic}.",
    )


def get_response(topic: str) -> str:
    """Build and run the LCEL chain for the given topic, returning the response."""
    provider = os.getenv("MODEL_PROVIDER", "openai")
    model_name = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
    api_key = os.getenv(f"{provider.upper()}_API_KEY")
    if not api_key:
        raise ValueError(f"Missing API key for provider '{provider}'. Set {provider.upper()}_API_KEY.")

    llm = init_chat_model(
        model=model_name,
        model_provider=provider,
        api_key=api_key,
        temperature=0.7,
    )
    prompt = build_prompt()
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"topic": topic})


def run_chain(topic: str) -> None:
    """Run a basic chain for the given topic and print the response."""
    response = get_response(topic)
    print(response)


def run_example() -> None:
    """Run a sample chain and print the response clearly."""
    topic = "space"
    print(f"--- Fun fact about {topic} ---")
    run_chain(topic)
    print("-----------------------------")


def main() -> None:
    """Run the example."""
    run_example()


if __name__ == "__main__":
    main()
