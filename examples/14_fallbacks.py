"""
Example: Fallback Chain with Logging

Demonstrates how to configure a fallback chain that uses a different model
configuration when the primary model fails due to a rate-limit error. The
fallback event is logged via the standard logging module.
"""

import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    """Raised when the model API rate limit is exceeded."""


# Primary model that raises a rate-limit error for demonstration purposes.
def primary_model(_messages):
    raise RateLimitError("Rate limit exceeded")


# Fallback model that handles rate-limit errors by retrying with a different
# model configuration.
def fallback_model(_messages):
    logger.info(
        "Fallback event: rate limit exceeded, retrying with a different model configuration"
    )
    return "This response came from the fallback model with a different configuration."


primary = RunnableLambda(primary_model)
fallback = RunnableLambda(fallback_model)

# Build a model with fallback support that only catches rate-limit errors.
model = primary.with_fallbacks([fallback], exceptions=(RateLimitError,))

# Create a simple chain: prompt -> model -> output parser.
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
chain = prompt | model | StrOutputParser()


if __name__ == "__main__":
    result = chain.invoke({"topic": "programming"})
    print(result)
