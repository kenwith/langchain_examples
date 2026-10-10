"""
Example: Fallback Chain with Logging

Demonstrates how to configure a fallback chain that uses a different model
when the primary model fails. The fallback event is logged via the standard
logging module.
"""

import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Primary model that always fails for demonstration purposes.
def primary_model(_messages):
    raise RuntimeError("Primary model is unavailable")


# Fallback model that handles the request when the primary model fails.
def fallback_model(_messages):
    logger.info("Fallback event: primary model failed, using fallback model")
    return "This response came from the fallback model."


primary = RunnableLambda(primary_model)
fallback = RunnableLambda(fallback_model)

# Build a model with fallback support.
model = primary.with_fallbacks([fallback])

# Create a simple chain: prompt -> model -> output parser.
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
chain = prompt | model | StrOutputParser()


if __name__ == "__main__":
    result = chain.invoke({"topic": "programming"})
    print(result)
