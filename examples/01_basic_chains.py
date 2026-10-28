"""Basic chain example.

This module demonstrates a simple LLMChain with a prompt template.
The flow:
1. Define a prompt template.
2. Create an LLM (e.g., OpenAI).
3. Build a chain from the prompt and LLM.
4. Run the chain with input variables.
5. Print the response.
"""

import os
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


def main():
    """Execute the basic chain example."""
    # Ensure API key is set
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    # Define prompt template
    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )

    # Initialize LLM
    llm = OpenAI(temperature=0.9)

    # Create chain
    chain = LLMChain(llm=llm, prompt=prompt)

    # Run the chain
    result = chain.run("eco-friendly water bottles")
    print(result)


if __name__ == "__main__":
    main()
