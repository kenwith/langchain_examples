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


def build_chain_with_fallback():
    """Build a runnable chain with retries and fallbacks from a list."""
    model_names = get_model_list()
    if not model_names:
        raise ValueError("MODELS environment variable must contain at least one model name")

    # Create the primary model with retry and failure logging
    primary_model = create_model(model_names[0]).with_retry(stop_after_attempt=2)
    primary_model = log_primary_failure(primary_model)
    primary_chain = PROMPT | primary_model | StrOutputParser()

    # Create fallback chains (if any) with retry and logging wrapper
    fallback_chains = []
    for name in model_names[1:]:
        model = create_model(name).with_retry(stop_after_attempt=2)
        fallback_model = log_fallback(model, name)
        fallback_chains.append(PROMPT | fallback_model | StrOutputParser())

    # If no fallbacks, just return the primary chain
    if not fallback_chains:
        return primary_chain
    return primary_chain.with_fallbacks(fallback_chains)


def main():
    """Run a simple demo with the fallback-enabled chain."""
