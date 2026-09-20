"""Basic chain example.

This script demonstrates how to build a simple prompt -> model -> output
parser chain using LangChain.

Usage:
    Set the LANGCHAIN_MODEL environment variable to the model you want to
    use (e.g. gpt-4o-mini, claude-3-5-sonnet, gemini-1.5-pro) and ensure
    the corresponding API key is set (e.g. OPENAI_API_KEY,
    ANTHROPIC_API_KEY, GOOGLE_API_KEY). Then run:

        python examples/01_basic_chains.py

    Alternatively, call build_chain(model_name="gpt-4o-mini") from your
    own code.

Sample output (will vary by model):
    LangChain is a framework for developing applications powered by
    language models. It provides standard interfaces for chains, agents,
    and retrieval, as well as integrations with other tools.
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


def run_example():
    """Build the chain and run it with a sample question."""
    chain = build_chain()
    response = chain.invoke({"question": "What is LangChain?"})
    print(response)


if __name__ == "__main__":
    run_example()
