"""Basic chain example.

This module demonstrates a simple LLMChain with a prompt template.
The flow:
1. Define a prompt template.
2. Create an LLM (e.g., OpenAI).
3. Build a chain from the prompt and LLM.
4. Run the chain with input variables.
5. Print the response.

Provider setup:
This example uses OpenAI as the LLM provider. The provider requires an API key
to be available in the environment. The key must be stored in the
OPENAI_API_KEY environment variable before running the module. The API key is
never hardcoded; it is read from the environment at runtime. Alternative
providers can be used by replacing the OpenAI LLM initialization with another
LangChain LLM wrapper and providing the corresponding credentials via
environment variables.
"""

import os
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


def build_chain() -> LLMChain:
    """Build and return the LLM chain used in this example."""
    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )
    llm = OpenAI(temperature=0.9)
    return LLMChain(llm=llm, prompt=prompt)


def run_example(product: str) -> str:
    """Run the prompt chain for a given product and return the response."""
    chain = build_chain()
    return chain.run(product)


def main():
    """Execute the basic chain example."""
    # Ensure API key is set
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    result = run_example("eco-friendly water bottles")
    print(result)


if __name__ == "__main__":
    main()
