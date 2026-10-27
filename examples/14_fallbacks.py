"""
Example: Fallback Chain with Logging and Retry

Demonstrates how to configure a fallback chain that retries once after a
rate-limit error and then uses a different model configuration if the retry
also fails. The fallback event is logged via the standard logging module.
"""

import logging
import time
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    """Raised when the model API rate limit is exceeded."""

    def __init__(self, message, retry_after=1.0):
        super().__init__(message)
        self.retry_after = retry_after


# Primary model that raises a rate-limit error for demonstration purposes.
def primary_model(_messages):
    raise RateLimitError("Rate limit exceeded", retry_after=0.5)


# Retry wrapper that waits for the specified retry_after period and then
# attempts the primary model once more.
def retry_model(messages):
    try:
        return primary_model(messages)
    except RateLimitError as e:
        logger.info(
            "Rate limit exceeded, waiting %.2f seconds before retry",
            e.retry_after,
        )
        time.sleep(e.retry_after)
        return primary_model(messages)


# Fallback model that handles rate-limit errors by retrying with a different
# model configuration.
def fallback_model(_messages):
    logger.info(
        "Fallback event: rate limit still exceeded, retrying with a different model configuration"
    )
    return "This response came from the fallback model with a different configuration."


primary = RunnableLambda(retry_model)
fallback = RunnableLambda(fallback_model)

# Build a model with fallback support that only catches rate-limit errors.
model = primary.with_fallbacks([fallback], exceptions=(RateLimitError,))

# Create a simple chain: prompt -> model -> output parser.
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
chain = prompt | model | StrOutputParser()


if __name__ == "__main__":
    result = chain.invoke({"topic": "programming"})
    print(result)
