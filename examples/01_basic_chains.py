"""Basic chain example.

This module demonstrates the simplest usage of LangChain: creating an LLMChain.
The chain flow is as follows:
1. Initialize a language model (OpenAI in this case) with a temperature setting.
2. Define a PromptTemplate that specifies the input variables and the template string.
3. Combine the LLM and prompt into an LLMChain.
4. Run the chain by passing a value for the input variable (e.g., a topic).
5. The chain formats the prompt, sends it to the LLM, and returns the response.

This script provides the following functions:
- build_prompt(): returns a PromptTemplate for the chain.
- run_chain(topic): runs the chain for a given topic and prints the raw response.
- run_example(): runs a sample topic and prints a clear, labeled output.
- main(): entry point that calls run_example().
"""

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


def build_prompt() -> PromptTemplate:
    """Build the prompt template for the chain."""
    return PromptTemplate(
        input_variables=["topic"],
        template="Tell me a fun fact about {topic}.",
    )


def run_chain(topic: str) -> None:
    """Run a basic chain for the given topic."""
    llm = OpenAI(temperature=0.7)
    prompt = build_prompt()
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(topic)
    print(response)


def run_example() -> None:
    """Run a sample chain and print the response clearly."""
    topic = "space"
    print(f"--- Fun fact about {topic} ---")
    run_chain(topic)
    print("-----------------------------")


def main() -> None:
    """Run the example."""
    run_example()


if __name__ == "__main__":
    main()
