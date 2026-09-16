"""
# 14. Fallbacks

| # | Example | Description |
|---|---------|-------------|
| 14 | fallbacks | Use init_chat_model with fallback and retry when primary fails |

This example demonstrates provider-agnostic model initialization with a list of
fallback models and a retry helper. It uses `init_chat_model` to create models
from a comma-separated list of model names (env var `MODELS`). The first model
is the primary, the rest are fallbacks. Each model is wrapped with `with_retry`,
and the chain uses `with_fallbacks`. A warning is logged whenever a fallback is
triggered. Credentials are read from environment variables.
"""

import logging
import os
import sys

from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableLambda

# Set up logging
logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
logger = logging.getLogger(__name__)


def create_model(model_name):
    """Create a chat model from a model name string."""
    return init_chat_model(model_name, temperature=0)


def get_model_list():
    """Read the model list from the MODELS environment variable."""
    models = os.getenv("MODELS", "openai/gpt-4o-mini,anthropic/claude-3-haiku-20240307")
    return [m.strip() for m in models.split(",") if m.strip()]


def log_fallback(model, fallback_name):
    """Wrap a fallback model to log a warning when it is invoked."""
    def invoke_with_log(input):
        logger.warning("Primary model failed; using fallback: %s", fallback_name)
        return model.invoke(input)
    return RunnableLambda(invoke_with_log)


def build_model_with_fallback():
    """Build a runnable model with retries and fallbacks from a list."""
    model_names = get_model_list()
    if not model_names:
        raise ValueError("MODELS environment variable must contain at least one model name")

    # Create the primary model with retry
    primary = create_model(model_names[0]).with_retry(stop_after_attempt=2)

    # Create fallback models (if any) with retry and logging wrapper
    fallbacks = []
    for name in model_names[1:]:
        model = create_model(name).with_retry(stop_after_attempt=2)
        fallbacks.append(log_fallback(model, name))

    # If no fallbacks, just return the primary
    if not fallbacks:
        return primary
    return primary.with_fallbacks(fallbacks)


def main():
    """Run a simple demo with the fallback-enabled model."""
    model = build_model_with_fallback()
    response = model.invoke("Hello, world!")
    print(response.content)


if __name__ == "__main__":
    main()
