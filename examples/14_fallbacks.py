"""
# 14. Fallbacks

| # | Example | Description |
|---|---------|-------------|
| 14 | fallbacks | Use init_chat_model with fallback and retry when primary fails |

This example demonstrates provider-agnostic model initialization with a fallback
model and a retry helper. It uses `init_chat_model` to create a primary model and
a fallback model, then combines them with `with_fallbacks` and `with_retry`.
Credentials are read from environment variables.
"""

import os

from langchain.chat_models import init_chat_model


def create_primary_model():
    """Create the primary chat model from environment variables."""
    return init_chat_model(
        os.getenv("PRIMARY_MODEL", "openai/gpt-4o-mini"),
        temperature=0,
    )


def create_fallback_model():
    """Create the fallback chat model from environment variables."""
    return init_chat_model(
        os.getenv("FALLBACK_MODEL", "anthropic/claude-3-haiku-20240307"),
        temperature=0,
    )


def build_model_with_fallback():
    """Build a runnable model with retries on both primary and fallback."""
    primary = create_primary_model().with_retry(stop_after_attempt=2)
    fallback = create_fallback_model().with_retry(stop_after_attempt=2)
    return primary.with_fallbacks([fallback])


def main():
    """Run a simple demo with the fallback-enabled model."""
    model = build_model_with_fallback()
    response = model.invoke("Hello, world!")
    print(response.content)


if __name__ == "__main__":
    main()
