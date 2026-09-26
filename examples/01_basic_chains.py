"""Basic chain example with provider-agnostic setup.

This module demonstrates the simplest usage of LangChain: creating an LLMChain.
The chain flow is as follows:
1. Initialize a language model using `init_chat_model`, which supports multiple providers
   (e.g., OpenAI, Anthropic) without changing the core logic. The provider and model name
   are read from environment variables, defaulting to OpenAI's GPT-3.5 Turbo.
2. Define a PromptTemplate that specifies the input variables and the template string.
3. Combine the LLM and prompt into an LLMChain.
4. Run the chain by passing a value for the input variable (e.g., a topic).
5. The chain formats the prompt, sends it to the LLM, and returns the response.

Environment variables:
- MODEL_PROVIDER: (optional) The provider name, e.g., "openai" or "anthropic". Defaults to "openai".
- MODEL_NAME: (optional) The model identifier, e.g., "gpt-3.5-turbo". Defaults to "gpt-3.5-turbo".
- {PROVIDER}_API_KEY: The API key for the selected provider (e.g., OPENAI_API_KEY). Must be set.

This script provides the following functions:
- build_prompt(): returns a PromptTemplate for the chain.
- run_chain(topic): runs the chain for a given topic and prints the raw response.
- run_example(): runs a sample topic and prints a clear, labeled output.
- main(): entry point that calls run_example().
"""

import os
from langchain.chat_models import init_chat_model
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


def build_prompt() -> PromptTemplate:
    """Build the prompt template for the chain."""
    return PromptTemplate(
        input_variables=["topic"],
        template="Tell me a fun fact about {topic}.",
    )


def run_chain(topic: str) -> None:
    """Run a basic chain for the given topic."""
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
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(topic)
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
