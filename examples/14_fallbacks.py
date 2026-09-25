"""
# 14. Fallbacks

| # | Example | Description |
|---|---------|-------------|
| 14 | fallbacks | Use init_chat_model with fallback and retry when primary fails |

This example demonstrates provider-agnostic model initialization with a list of
fallback models and a retry helper. It uses `init_chat_model` to create models
from a comma-separated list of model names (env var `MODELS`). The first model
is the primary, the rest are fallbacks. Each model is wrapped with `with_retry`,
and the chain uses `with_fallbacks`. A warning is logged when the primary model
fails and another warning is logged when a fallback is triggered. Credentials
are read from environment variables.

The `fallback_chain` helper builds a runnable chain from a primary model name
and a list of fallback model names, with clear error handling for model
initialization failures.
"""

import logging
import os
import sys

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

# Set up logging
logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
logger = logging.getLogger(__name__)

PROMPT = ChatPromptTemplate.from_template("Tell me a short joke about {topic}")


def create_model(model_name):
    """Create a chat model from a model name string."""
    return init_chat_model(model_name, temperature=0)


def get_model_list():
    """Read the model list from the MODELS environment variable."""
    models = os.getenv("MODELS", "openai/gpt-4o-mini,anthropic/claude-3-haiku-20240307")
    return [m.strip() for m in models.split(",") if m.strip()]


def log_primary_failure(model):
    """Wrap the primary model to log when it fails."""
    def invoke_with_log(input):
        try:
            return model.invoke(input)
        except Exception as e:
            logger.warning("Primary model failed: %s", e)
            raise
    return RunnableLambda(invoke_with_log)


def log_fallback(model, fallback_name):
    """Wrap a fallback model to log a warning when it is invoked."""
    def invoke_with_log(input):
        logger.warning("Fallback triggered: %s", fallback_name)
        return model.invoke(input)
    return RunnableLambda(invoke_with_log)


def fallback_chain(primary_model_name, fallback_model_names):
    """Build a runnable chain with retries and fallbacks from model names.

    Args:
        primary_model_name: The primary model name.
        fallback_model_names: A list of fallback model names.

    Returns:
        A runnable chain that tries the primary model, then fallbacks.
    """
    try:
        primary_model = create_model(primary_model_name)
    except Exception as e:
        raise ValueError(
            f"Failed to create primary model '{primary_model_name}': {e}"
        ) from e

    fallback_models = []
    for name in fallback_model_names:
        try:
            model = create_model(name)
        except Exception as e:
            logger.warning("Failed to create fallback model '%s': %s", name, e)
            continue
        fallback_models.append((model, name))

    if not fallback_models:
        logger.warning("No valid fallback models configured; using primary only.")

    primary_model = primary_model.with_retry(stop_after_attempt=2)
    primary_model = log_primary_failure(primary_model)
    primary_chain = PROMPT | primary_model | StrOutputParser()

    fallback_chains = []
    for model, name in fallback_models:
        model = model.with_retry(stop_after_attempt=2)
        fallback_model = log_fallback(model, name)
        fallback_chains.append(PROMPT | fallback_model | StrOutputParser())

    if not fallback_chains:
        return primary_chain
    return primary_chain.with_fallbacks(fallback_chains)


def build_chain_with_fallback():
    """Build a runnable chain from the MODELS environment variable."""
    model_names = get_model_list()
    if not model_names:
        raise ValueError("MODELS environment variable must contain at least one model name")
    return fallback_chain(model_names[0], model_names[1:])


def main():
    """Run a simple demo with the fallback-enabled chain."""
    topic = sys.argv[1] if len(sys.argv) > 1 else "programming"
    try:
        chain = build_chain_with_fallback()
        response = chain.invoke({"topic": topic})
        print(response)
    except Exception as e:
        logger.error("Chain execution failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
