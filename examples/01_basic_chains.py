"""Basic chain example.

This script demonstrates how to build a simple prompt -> model -> output
parser chain using LangChain.

Chain structure
---------------
The chain is composed of three components connected with the ``|`` operator:

    prompt = ChatPromptTemplate.from_template(...)
    model = init_chat_model(...)
    parser = StrOutputParser()
    chain = prompt | model | parser

- ``prompt``: A ``ChatPromptTemplate`` that takes a dictionary with a
  ``"question"`` key and formats it into a chat message.
- ``model``: A chat model initialized via ``init_chat_model()``. It receives
  the formatted prompt and returns an ``AIMessage``.
- ``parser``: A ``StrOutputParser`` that extracts the text content from the
  model's output, so the final result is a plain string.

The resulting ``chain`` is a ``Runnable`` that accepts a dictionary with a
``"question"`` key and returns a string. This is the standard way to compose
LangChain components: each ``|`` passes the output of the left-hand side as
the input to the right-hand side.

Why `init_chat_model`?
    `init_chat_model()` provides a unified interface for initializing chat
    models from different providers. This keeps the example provider-agnostic
    and consistent with other LangChain examples, while still allowing the
    model to be selected at runtime.

Model selection:
    The model is selected by setting the LANGCHAIN_MODEL environment variable
    or by passing a `model_name` to `build_chain()`. The corresponding API key
    must be available in the environment (e.g. OPENAI_API_KEY, ANTHROPIC_API_KEY,
    GOOGLE_API_KEY). Supported models include:
        - OpenAI: gpt-4o-mini
        - Anthropic: claude-3-5-sonnet
        - Google: gemini-1.5-pro

Usage:
    1. Install LangChain and the provider SDK for the model you want to use.
       For example, for OpenAI:

           pip install langchain langchain-openai

    2. Set the model and API key environment variables:

           export LANGCHAIN_MODEL=gpt-4o-mini
           export OPENAI_API_KEY=your-api-key

    3. Run the example:

           python examples/01_basic_chains.py

    Alternatively, call ``build_chain(model_name="gpt-4o-mini")`` from your
    own code and invoke the returned chain:

        chain = build_chain("gpt-4o-mini")
        response = chain.invoke({"question": "What is LangChain?"})

Input/Output:
    The chain expects a dictionary with a single ``"question"`` key whose
    value is a string. It returns the model's answer as a string. The
    ``format_response`` helper is provided only to make the printed output
    easier to read; it is not part of the chain itself.

Sample output (will vary by model):
    LangChain is a framework for developing applications powered by
    language models. It provides standard interfaces for chains, agents,
    and retrieval, as well as integrations with other tools.
"""

import os
import textwrap
from typing import Optional

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable


def build_chain(model_name: Optional[str] = None) -> Runnable[dict, str]:
    """Build a basic prompt -> model -> output parser chain.

    Args:
        model_name: Optional model name. If not provided, the model is
            loaded from the LANGCHAIN_MODEL environment variable.

    Returns:
        A runnable chain that accepts a dictionary with a "question" key
        and returns the model's text response.

    The chain is built by piping three components together:

        prompt = ChatPromptTemplate.from_template(...)
        model = init_chat_model(...)
        chain = prompt | model | StrOutputParser()

    The ``|`` operator composes runnables: the prompt formats the input
    dictionary into a chat message, the model generates an ``AIMessage``,
    and the output parser converts it to a string. This means the returned
    chain can be invoked directly with ``chain.invoke({"question": "..."})``.

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


def format_response(response: str, line_length: int = 80) -> str:
    """Format the model response for better readability.

    Strips leading/trailing whitespace, normalizes line endings, wraps
    each paragraph to `line_length`, and ensures paragraphs are separated
    by a single blank line.

    Args:
        response: Raw response string from the chain.
        line_length: Maximum line length for wrapping.

    Returns:
        A cleaned and wrapped version of the response.
    """
    paragraphs = [
        " ".join(paragraph.split())
        for paragraph in response.strip().split("\n\n")
        if paragraph.strip()
    ]
    wrapped_paragraphs = [
        textwrap.fill(paragraph, width=line_length)
        for paragraph in paragraphs
    ]
    return "\n\n".join(wrapped_paragraphs)


def run_example() -> None:
    """Build the chain and run it with a sample question."""
    chain: Runnable[dict, str] = build_chain()
    response: str = chain.invoke({"question": "What is LangChain?"})
    print(format_response(response))


if __name__ == "__main__":
    run_example()
