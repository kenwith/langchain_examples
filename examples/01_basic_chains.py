"""
Basic LangChain Chains Example
==============================

This module demonstrates how to build and run simple language model chains
using LangChain's expression language (LCEL). It is designed to be both a
standalone script and a reusable module for learning and experimentation.

Features demonstrated
---------------------

- Creating reusable prompt templates with `build_prompt_template`.
- Initializing a chat model from environment variables or a provided name.
- Running chains that return plain text, structured JSON, and fallback models.
- Composing sequential chains with intermediate value passing.
- Graceful handling of missing or invalid API keys via `run_prompt`.

Module Functions
----------------
- `get_model` — Initialize a chat model from environment variables or a name.
- `build_prompt_template` — Construct a reusable `ChatPromptTemplate`.
- `run_chain` — Build and invoke a prompt → model → parser pipeline.
- `run_prompt` — Run a prompt with error handling and pretty-printed output.
- `format_output` — Format strings, dicts, and lists for console display.
- `basic_string_chain` — Simple free-text generation example.
- `structured_output_chain` — Pydantic-validated JSON output example.
- `chain_with_fallback` — Model fallback resilience example.
- `sequential_chains` — Multi‑step chain composition example.
- `main` — Entry point for the all the examples.

Usage Notes
-----------
Before running this script, ensure you have set the appropriate API keys
in your environment. The default model is `openai/gpt-4o-mini`, which
requires an OpenAI API key. You can set it as:

    export OPENAI_API_KEY="your-api-key"

Alternatively, you can set the `LANGCHAIN_MODEL` environment variable to
use a different model provider (e.g., Anthropic, Google, etc.) and set the
corresponding API key.

Run the script directly to execute all examples:

    python examples/01_basic_chains.py

You can also import the module and call `main()` programmatically:

    from examples.01_basic_chains import main
    main()

Each example function can also be called individually, allowing you to
integrate specific chain patterns into your own code.
"""

import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import BaseOutputParser, JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda

from pydantic import BaseModel, Field

load_dotenv()


def get_model(model_name: Optional[str] = None) -> BaseChatModel:
    """Initialize a chat model from environment or provided name.

    Args:
        model_name: Optional model identifier. If not provided, reads from
            LANGCHAIN_MODEL environment variable, defaulting to "openai/gpt-4o-mini".

    Returns:
        An initialized BaseChatModel instance ready for use in chains.
    """
    name = model_name or os.getenv("LANGCHAIN_MODEL", "openai/gpt-4o-mini")
    return init_chat_model(name)


def build_prompt_template(
    system_message: str,
    user_message: str,
    input_variables: Optional[List[str]] = None,
) -> ChatPromptTemplate:
    """Build a reusable ChatPromptTemplate with system and user messages.

    This helper creates consistent prompt templates across chains, reducing
    duplication and ensuring standardized formatting.

    Args:
        system_message: The system prompt text, can contain {variable} placeholders.
        user_message: The user prompt text, can contain {variable} placeholders.
        input_variables: Optional list of variable names expected by the template.
            If not provided, variables are inferred from the message templates.

    Returns:
        A configured ChatPromptTemplate ready for use in LLM chains.

    Example:
        >>> prompt = build_prompt_template(
        ...     system_message="You are a {role}.",
        ...     user_message="Explain {topic} to a {audience}.",
        ...     input_variables=["role", "topic", "audience"]
        ... )
    """
    messages = [
        ("system", system_message),
        ("user", user_message),
    ]
    if input_variables is not None:
        return ChatPromptTemplate.from_messages(messages).partial(
            **{var: "{" + var + "}" for var in input_variables}
        )
    return ChatPromptTemplate.from_messages(messages)


def run_chain(
    prompt: ChatPromptTemplate,
    inputs: Dict[str, Any],
    model_name: Optional[str] = None,
    parser: Optional[BaseOutputParser] = None,
) -> Any:
    """Load the model and run the given prompt with the provided inputs.

    This is a convenience helper that builds a chain from a prompt, a model,
    and an optional output parser, then invokes it with the given inputs.

    Args:
        prompt: The ChatPromptTemplate to use.
        inputs: A dictionary of input variables for the prompt.
        model_name: Optional model name to pass to `get_model`.
        parser: Optional output parser. Defaults to StrOutputParser.

    Returns:
        The output from the chain, whose type depends on the parser used.
    """
    model = get_model(model_name)
    if parser is None:
        parser = StrOutputParser()
    chain = prompt | model | parser
    return chain.invoke(inputs)


def format_output(output: Any, title: str = "Result") -> str:
    """Format the model output for clear display.

    This helper parses the raw output from a chain and returns a formatted
    string suitable for console printing. It handles strings, dictionaries,
    and lists gracefully.

    Args:
        output: The raw output from the chain (string, dict, or list).
        title: A title for the output section.

    Returns:
        A formatted string representation of the output.
    """
    if isinstance(output, dict):
        lines = [f"{title}:"]
        for key, value in output.items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)
    elif isinstance(output, list):
        lines = [f"{title}:"]
        for item in output:
            lines.append(f"  - {item}")
        return "\n".join(lines)
    else:
        return f"{title}: {output}"


def run_prompt(
    prompt: ChatPromptTemplate,
    inputs: Dict[str, Any],
    model_name: Optional[str] = None,
) -> Any:
    """Execute a single prompt, print the response, and handle missing API keys gracefully.

    This is a convenience wrapper around `run_chain` that prints the result and
    catches common exceptions related to missing or invalid API keys. If an error
    occurs, it prints a user-friendly message and returns None.

    Args:
        prompt: The ChatPromptTemplate to use.
        inputs: Dictionary of input variables for the prompt.
        model_name: Optional model name to pass to `get_model`.

    Returns:
        The output from the chain, or None if an error occurred.
    """
    try:
        result = run_chain(prompt, inputs, model_name)
        print(format_output(result, "Result"))
        return result
    except Exception as e:
        error_str = str(e).lower()
        if "api_key" in error_str or "api key" in error_str or "auth" in error_str:
            print("Error: Missing or invalid API key. Please set the appropriate environment variable (e.g., OPENAI_API_KEY).")
        else:
            print(f"Error running prompt: {e}")
        return None


def basic_string_chain() -> str:
    """Run a simple chain: prompt -> LLM -> string output.

    Returns:
        The generated explanation as a string.
    """
    print("=== Basic String Chain ===")

    prompt = build_prompt_template(
        system_message="You are a helpful assistant that explains concepts simply.",
        user_message="Explain {topic} in 2-3 sentences for a {audience}.",
        input_variables=["topic", "audience"],
    )

    result = run_chain(
        prompt,
        {"topic": "quantum computing", "audience": "10-year-old"},
    )
    print(format_output(result, "Result") + "\n")
    return result


def structured_output_chain() -> Dict[str, Any]:
    """Run a chain with structured JSON output using Pydantic validation.

    Returns:
        A dictionary containing the structured explanation with keys:
        concept, summary, key_points, and difficulty.
    """
    print("=== Structured Output Chain ===")

    class Explanation(BaseModel):
        """Structured explanation output schema."""

        concept: str = Field(description="The concept being explained")
        summary: str = Field(description="2-3 sentence summary")
        key_points: List[str] = Field(description="3 key points")
        difficulty: str = Field(description="beginner, intermediate, or advanced")

    prompt = build_prompt_template(
        system_message="Explain the concept and return structured JSON.",
        user_message="Explain {topic} for a {audience}. Return JSON matching the schema.",
        input_variables=["topic", "audience"],
    )

    parser = JsonOutputParser(pydantic_object=Explanation)
    result = run_chain(
        prompt,
        {"topic": "neural networks", "audience": "college student"},
        parser=parser,
    )
    print(format_output(result, "Result") + "\n")
    return result


def chain_with_fallback() -> str:
    """Run a chain with a fallback model for resilience.

    Returns:
        The model's response as a string.
    """
    print("=== Chain with Fallback ===")

    prompt = build_prompt_template(
        system_message="You are a helpful assistant.",
        user_message="{question}",
        input_variables=["question"],
    )

    primary_model = get_model()
    fallback_model = init_chat_model("openai/gpt-4o-mini")
    model = primary_model.with_fallbacks([fallback_model])

    chain: Runnable[Dict[str, Any], str] = prompt | model | StrOutputParser()
    result = chain.invoke({"question": "What is the capital of France?"})
    print(format_output(result, "Result") + "\n")
    return result


def sequential_chains() -> Dict[str, str]:
    """Run multiple chains in sequence to generate a blog post structure.

    Returns:
        A dictionary containing the generated topic, outline, and intro.
    """
    print("=== Sequential Chains ===")

    topic_prompt = build_prompt_template(
        system_message="You are a creative blog topic generator.",
        user_message="Suggest one interesting topic about {domain} for a blog post.",
        input_variables=["domain"],
    )
    outline_prompt = build_prompt_template(
        system_message="You are an expert content strategist.",
        user_message="Create a 3-point outline for a blog post about: {topic}",
        input_variables=["topic"],
    )
    intro_prompt = build_prompt_template(
        system_message="You are an engaging blog writer.",
        user_message="Write an engaging intro paragraph for a blog post with this outline:\n{outline}",
        input_variables=["outline"],
    )

    # Combined chain using RunnableLambda for intermediate value passing.
    full_chain: Runnable[Dict[str, Any], Dict[str, str]] = (
        RunnableLambda(
            lambda x: {
                "topic": run_chain(topic_prompt, {"domain": x["domain"]}),
                "domain": x["domain"],
            }
        )
        | RunnableLambda(
            lambda x: {
                "outline": run_chain(outline_prompt, {"topic": x["topic"]}),
                "topic": x["topic"],
            }
        )
        | RunnableLambda(
            lambda x: {
                "intro": run_chain(intro_prompt, {"outline": x["outline"]}),
                "outline": x["outline"],
                "topic": x["topic"],
            }
        )
    )

    result = full_chain.invoke({"domain": "artificial intelligence"})
    print(format_output(result["topic"], "Topic"))
    print(format_output(result["outline"], "Outline"))
    print(format_output(result["intro"], "Intro") + "\n")
    return result


def main() -> None:
    """Run all the example chains.

    This is the main entry point for the script. It runs the `run_prompt`
    demo first (to show graceful error handling), followed by the four
    chain configuration examples. After completion it prints a summary
    message.
    """
    # Demonstrate run_prompt with graceful error handling
    print("=== Using run_prompt helper ===")
    prompt = build_prompt_template(
        system_message="You are a helpful assistant.",
        user_message="What is the capital of {country}?",
        input_variables=["country"],
    )
    run_prompt(prompt, {"country": "France"})

    # Run the other examples
    basic_string_chain()
    structured_output_chain()
    chain_with_fallback()
    sequential_chains()

    print("All basic chain examples completed!")


if __name__ == "__main__":
    main()
