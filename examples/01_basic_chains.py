"""Basic chains example.

This module demonstrates how to build and run simple chains with LangChain.
It uses an LLMChain to generate a company name from a product description and
then a SimpleSequentialChain to create a catchphrase for that company name.

Run this module directly to see the example output.
"""

from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate


def main() -> None:
    """Run the basic chains example."""
    llm = OpenAI(temperature=0.7)

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


if __name__ == "__main__":
    main()
