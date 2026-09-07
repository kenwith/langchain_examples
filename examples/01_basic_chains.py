"""
Basic LangChain Chains Example
==============================

This module demonstrates how to build and run simple language model chains
using LangChain's expression language (LCEL). It covers:

- Creating reusable prompt templates with `build_prompt_template`.
- Initializing a chat model from environment variables or a provided name.
- Running chains that return plain text, structured JSON, and fallback models.
- Composing sequential chains with intermediate value passing.

The module is provider-agnostic: it uses `init_chat_model` to load a model
based on the `LANGCHAIN_MODEL` environment variable (default: "openai/gpt-4o-mini").
Make sure your API keys are set in the environment (e.g., `OPENAI_API_KEY`).

Example usage:
    python examples/01_basic_chains.py

This will execute all the chain examples and print their outputs.
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
    print(f"Result: {result}\n")
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
    print(f"Result: {result}\n")
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
    print(f"Result: {result}\n")
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

    # Combined chain using RunnableLambda for intermediate value passing,
    # with run_chain used inside each lambda to keep the logic concise.
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
    print(f"Topic: {result['topic']}")
    print(f"Outline: {result['outline']}")
    print(f"Intro: {result['intro']}\n")
    return result


if __name__ == "__main__":
    basic_string_chain()
    structured_output_chain()
    chain_with_fallback()
    sequential_chains()
    print("All basic chain examples completed!")
