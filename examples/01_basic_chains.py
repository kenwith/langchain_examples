"""Basic chain example using LangChain's provider-agnostic init_chat_model.

This module demonstrates how to build a simple prompt -> model -> output
parser chain. Instead of hardcoding a specific chat model class, we use
`init_chat_model()` which selects the appropriate implementation based on
the model name and available environment variables (e.g. OPENAI_API_KEY,
ANTHROPIC_API_KEY, GOOGLE_API_KEY). This makes it easy to switch providers
without changing the chain construction code.

Before running, set the LANGCHAIN_MODEL environment variable to the model
you want to use. For example:

    export LANGCHAIN_MODEL="gpt-4o-mini"

You also need the API key for the provider of that model (e.g.
OPENAI_API_KEY for OpenAI models).

Run this example with:
    python examples/01_basic_chains.py

Expected output:
The script prints a short answer to the question "What is LangChain?".
The exact wording depends on the model you choose, but it should be a
concise explanation similar to:

    LangChain is a framework for developing applications powered by
    language models.
"""

import os

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def build_chain(model_name=None):
    """Build a basic prompt -> model -> output parser chain.

    Args:
        model_name: Optional model name. If not provided, the model is
            loaded from the LANGCHAIN_MODEL environment variable.

    The model is initialized with `init_chat_model()`, which reads the
    appropriate API key from environment variables. To use a different
    provider, change the model name and ensure the corresponding
    environment variable is set (e.g. ANTHROPIC_API_KEY, GOOGLE_API_KEY).
    """
    if model_name is None:
        model_name = os.getenv("LANGCHAIN_MODEL")
    if not model_name:
        raise ValueError(
            "No model name provided. Pass a model_name to build_chain() "
            "or set the LANGCHAIN_MODEL environment variable. "
            "Supported models include 'gpt-4o-mini' for OpenAI, "
            "'claude-3-5-sonnet' for Anthropic, and 'gemini-1.5-pro' for "
            "Google. You also need the corresponding API key in your "
            "environment."
        )

    model = init_chat_model(
        model=model_name,
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Answer the following question:\n\n{question}"
    )

    chain = prompt | model | StrOutputParser()
    return chain


def run_chain():
    """Build the chain and run it with a sample question."""
    chain = build_chain()
    response = chain.invoke({"question": "What is LangChain?"})
    print(response)


if __name__ == "__main__":
    run_chain()
