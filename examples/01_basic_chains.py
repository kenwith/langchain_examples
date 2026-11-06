"""Basic chain example.

This module demonstrates a simple LLMChain with a prompt template.

A chain in LangChain is a sequence of components that processes an input and
produces an output. In this example, the chain combines a prompt template and
an LLM to generate a response. The prompt template defines the structure of the
input and the instruction, while the LLM provides the generative capability.
The chain orchestrates the data flow: it formats the prompt with the given
input, sends it to the LLM, and returns the model's output. This pattern allows
for reusable and composable building blocks, where each component can be
swapped or extended independently.

For a quick inline demonstration, the chain can be built and invoked directly:
`LLMChain(llm=OpenAI(), prompt=PromptTemplate(...)).run("eco-friendly water bottles")`
returns a generated company name. The full example below shows the recommended
structure with separate functions for clarity.

Chain flow:
1. Define a prompt template. The template declares the input variables and
   contains the instruction that will be sent to the language model.
2. Create an LLM instance, such as OpenAI, with optional settings like
   temperature. This instance represents the model that generates the response.
3. Build an LLMChain from the prompt and the LLM. The chain handles the
   formatting of the prompt and the invocation of the model.
4. Run the chain with the input variables. The chain fills the template with
   the provided values, sends the formatted prompt to the LLM, and returns the
   generated text.
5. Print the response. The example prints the generated company name to the
   console.

Provider configuration:
This example uses OpenAI as the LLM provider. The provider requires an API key
to be available in the environment. The key must be stored in the
OPENAI_API_KEY environment variable before running the module. The API key is
never hardcoded; it is read from the environment at runtime. For local
development, the variable can be exported in the shell or loaded from a .env
file by an external tool such as python-dotenv. Alternative providers can be
used by replacing the OpenAI LLM initialization in build_chain() with another
LangChain LLM wrapper and providing the corresponding credentials via
environment variables.

Expected output:
The example prints a generated company name for the given product. Because
LLM responses are non-deterministic, the exact output will vary. Example
outputs include "EcoBottle Co." or "GreenWave Bottles". The response is
printed as a single line to the console.

Reuse:
The build_chain() function constructs the LLMChain and can be reused in other
modules by importing this module. The run_example(product) function accepts a
product description and returns the model's suggestion as a string, allowing
callers to integrate the chain into larger workflows.

Import behavior:
The module is safe to import without side effects. No network requests, API
calls, or chain executions occur at import time. The LLM chain is only invoked
inside the main() function, which runs only when this file is executed directly
as a script. Therefore, importing this module in another program does not
require an API key and does not consume resources until a function is called.

Usage:
    Run the script directly to see the example in action:
        python examples/01_basic_chains.py
    The main() function can also be imported and called programmatically,
    but it requires the OPENAI_API_KEY environment variable to be set.

Usage example:
    The following snippet demonstrates how to use the module's functions
    programmatically. Because the filename begins with a digit, the module
    is loaded with importlib:

    import importlib
    basic_chains = importlib.import_module("examples.01_basic_chains")
    result = basic_chains.run_example("eco-friendly water bottles")
    print(result)

    The chain can also be constructed and invoked directly:

    from langchain.llms import OpenAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain

    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )
    llm = OpenAI(temperature=0.9)
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run("eco-friendly water bottles")
    print(response)

Customization:
    To use a different product description, modify the argument passed to
    run_example() inside main(). To switch to a different LLM provider,
    replace the OpenAI initialization in build_chain() with another LangChain
    LLM wrapper and ensure the corresponding credentials are available in the
    environment.
"""

import os
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


def build_chain() -> LLMChain:
    """Build and return the LLM chain used in this example."""
    # Define the prompt template. The template declares the input variable
    # "product" and contains the instruction that will be sent to the model.
    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )
    # Create an OpenAI LLM instance with a temperature setting. Temperature
    # controls the randomness of the model's output; higher values like 0.9
    # produce more creative responses.
    llm = OpenAI(temperature=0.9)
    # Build an LLMChain that combines the prompt and the LLM. The chain
    # handles formatting the prompt and invoking the model.
    return LLMChain(llm=llm, prompt=prompt)


def run_example(product: str) -> str:
    """Run the prompt chain for a given product and return the response."""
    # Build the chain once and reuse it for the specific product.
    chain = build_chain()
    # Run the chain with the provided product description. The chain fills
    # the template, sends it to the LLM, and returns the generated text.
    return chain.run(product)


def main():
    """Run the example and print the model's generated company name.

    The function verifies that the OPENAI_API_KEY environment variable is
    present, executes the chain for a sample product, and prints the response
    to standard output.
    """
    # Check that the API key is available in the environment. The key is
    # never hardcoded; it must be set before running the script.
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    # Execute the chain for a sample product and capture the result.
    result = run_example("eco-friendly water bottles")
    # Print the generated company name to the console.
    print(result)


if __name__ == "__main__":
    main()
