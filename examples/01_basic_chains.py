"""Basic chain example.

This example demonstrates a minimal LangChain chain: a prompt template,
a chat model, and an output parser. The chain is provider-agnostic: any
LangChain chat model can be used in place of ChatOpenAI.

The module exposes a reusable run_chain() function that accepts any
LangChain chat model, a prompt template, and template variables. It builds
a prompt, pipes it through the model and a string output parser, and
returns the parsed response. This keeps the core chain logic separate from
model configuration, so the same function can be reused with different
providers and prompts.

A format_response() helper is also provided. It wraps a raw model response
in a simple header and trailing newline, making terminal output easier to
read when running the example directly. The helper is intentionally small
and can be adapted to suit different output requirements.

Provider configuration:
- OpenAI (default): set the OPENAI_API_KEY environment variable, or create
  a .env file in the project root with:
  OPENAI_API_KEY=your-api-key
  The default model is "gpt-4o-mini". Override with OPENAI_MODEL if needed.
- Anthropic: install langchain-anthropic and set ANTHROPIC_API_KEY.
  Replace ChatOpenAI with ChatAnthropic(model="claude-3-5-haiku-latest").
- Google: install langchain-google-vertexai or langchain-google-genai and
  set GOOGLE_API_KEY or GOOGLE_APPLICATION_CREDENTIALS. Replace ChatOpenAI
  with ChatGoogleGenerativeAI(model="gemini-1.5-flash").
- Local models via Ollama: install langchain-ollama and run a local model
  with Ollama. Replace ChatOpenAI with ChatOllama(model="llama3.2").

Expected output:
A formatted response from the model, e.g.:

--- Response ---
Why did the chicken cross the road?
To get to the other side.
"""

import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


def run_chain(llm, prompt_template, variables):
    """Run a prompt through a chat model and return the response.

    Args:
        llm: A LangChain chat model instance.
        prompt_template: A string template with placeholders.
        variables: A dictionary of values for the template placeholders.

    Returns:
        The parsed string response from the model.
    """
    prompt = PromptTemplate.from_template(prompt_template)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke(variables)


def format_response(response):
    """Format a chain response for display.

    Adds a simple header and trailing newline to make the output easier to
    read when running the example from a terminal.

    Args:
        response: The raw string response from the chain.

    Returns:
        A formatted string with the response.
    """
    return f"--- Response ---\n{response}\n"


def run_example():
    """Run the basic chain example."""
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model, temperature=0.7)
    response = run_chain(llm, "Tell me a {adjective} joke.", {"adjective": "funny"})
    print(format_response(response))


if __name__ == "__main__":
    run_example()
