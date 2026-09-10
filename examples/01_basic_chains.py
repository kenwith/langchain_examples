"""Basic chain example using LangChain's provider-agnostic init_chat_model.

This module demonstrates how to build a simple prompt -> model -> output
parser chain. Instead of hardcoding a specific chat model class, we use
`init_chat_model()` which selects the appropriate implementation based on
the model name and available environment variables (e.g. OPENAI_API_KEY,
ANTHROPIC_API_KEY, GOOGLE_API_KEY). This makes it easy to switch providers
without changing the chain construction code.

Run this example with:
    python examples/01_basic_chains.py
"""

import os

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def build_chain():
    """Build a basic prompt -> model -> output parser chain.

    The model is initialized with `init_chat_model()`, which reads the
    appropriate API key from environment variables. To use a different
    provider, change the model name and ensure the corresponding
    environment variable is set (e.g. ANTHROPIC_API_KEY, GOOGLE_API_KEY).
    """
    model = init_chat_model(
        model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
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
