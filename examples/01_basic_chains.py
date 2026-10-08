"""Basic chains example.

This module demonstrates how to build and run simple chains with LangChain.
It uses an LLMChain to generate a company name from a product description and
then a SimpleSequentialChain to create a catchphrase for that company name.

The model initialization is provider-agnostic: the run_example function accepts
any LangChain LLM, so the same chain logic can be reused with different
providers (e.g., OpenAI, Cohere, Hugging Face) by passing a compatible instance.

Run this module directly to see the example output.
"""

from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate


def run_example(llm) -> None:
    """Run the basic chains example with the provided LLM instance."""
    first_prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )
    first_chain = LLMChain(llm=llm, prompt=first_prompt, verbose=True)

    second_prompt = PromptTemplate(
        input_variables=["company_name"],
        template="Write a catchphrase for the following company: {company_name}",
    )
    second_chain = LLMChain(llm=llm, prompt=second_prompt, verbose=True)

    overall_chain = SimpleSequentialChain(
        chains=[first_chain, second_chain],
        verbose=True,
    )

    product = "colorful socks"
    result = overall_chain.run(product)
    print(result)


def main() -> None:
    """Initialize a model and run the example."""
    llm = OpenAI(temperature=0.7)
    run_example(llm)


if __name__ == "__main__":
    main()
