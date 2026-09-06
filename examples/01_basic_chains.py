"""
Basic LangChain Chains Example

Demonstrates: LLM + Prompt + Output Parser
Provider-agnostic using init_chat_model
"""
import os
from typing import Any, Callable, Dict, List, Optional, Union

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnableSequence
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

    model = get_model()
    parser = StrOutputParser()

    chain: Runnable[Dict[str, Any], str] = prompt | model | parser

    result = chain.invoke({"topic": "quantum computing", "audience": "10-year-old"})
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

    model = get_model()
    parser = JsonOutputParser(pydantic_object=Explanation)

    chain: Runnable[Dict[str, Any], Dict[str, Any]] = prompt | model | parser

    result = chain.invoke({"topic": "neural networks", "audience": "college student"})
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

    chain: Runnable[Dict[str, Any], str] = (
        prompt | primary_model.with_fallbacks([fallback_model]) | StrOutputParser()
    )

    result = chain.invoke({"question": "What is the capital of France?"})
    print(f"Result: {result}\n")
    return result


def sequential_chains() -> Dict[str, str]:
    """Run multiple chains in sequence to generate a blog post structure.

    Returns:
        A dictionary containing the generated topic, outline, and intro.
    """
    print("=== Sequential Chains ===")

    model = get_model()

    # Chain 1: Generate a topic
    topic_prompt = build_prompt_template(
        system_message="You are a creative blog topic generator.",
        user_message="Suggest one interesting topic about {domain} for a blog post.",
        input_variables=["domain"],
    )
    topic_chain: Runnable[Dict[str, Any], str] = topic_prompt | model | StrOutputParser()

    # Chain 2: Create outline from topic
    outline_prompt = build_prompt_template(
        system_message="You are an expert content strategist.",
        user_message="Create a 3-point outline for a blog post about: {topic}",
        input_variables=["topic"],
    )
    outline_chain: Runnable[Dict[str, Any], str] = outline_prompt | model | StrOutputParser()

    # Chain 3: Write intro from outline
    intro_prompt = build_prompt_template(
        system_message="You are an engaging blog writer.",
        user_message="Write an engaging intro paragraph for a blog post with this outline:\n{outline}",
        input_variables=["outline"],
    )
    intro_chain: Runnable[Dict[str, Any], str] = intro_prompt | model | StrOutputParser()

    # Combined chain using RunnableLambda for intermediate value passing
    full_chain: Runnable[Dict[str, Any], Dict[str, str]] = (
        RunnableLambda(lambda x: {"topic": topic_chain.invoke(x), "domain": x["domain"]})
        | RunnableLambda(lambda x: {"outline": outline_chain.invoke({"topic": x["topic"]}), "topic": x["topic"]})
        | RunnableLambda(lambda x: {"intro": intro_chain.invoke({"outline": x["outline"]}), "outline": x["outline"], "topic": x["topic"]})
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
