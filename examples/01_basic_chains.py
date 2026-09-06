"""
Basic LangChain Chains Example

Demonstrates: LLM + Prompt + Output Parser
Provider-agnostic using init_chat_model
"""
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field

load_dotenv()


def get_model():
    """Initialize model from environment (LANGCHAIN_MODEL)"""
    model_name = os.getenv("LANGCHAIN_MODEL", "openai/gpt-4o-mini")
    return init_chat_model(model_name)


def basic_string_chain():
    """Simple chain: prompt -> LLM -> string output"""
    print("=== Basic String Chain ===")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that explains concepts simply."),
        ("user", "Explain {topic} in 2-3 sentences for a {audience}."),
    ])

    model = get_model()
    parser = StrOutputParser()

    chain = prompt | model | parser

    result = chain.invoke({"topic": "quantum computing", "audience": "10-year-old"})
    print(f"Result: {result}\n")
    return result


def structured_output_chain():
    """Chain with structured JSON output using Pydantic"""
    print("=== Structured Output Chain ===")

    class Explanation(BaseModel):
        concept: str = Field(description="The concept being explained")
        summary: str = Field(description="2-3 sentence summary")
        key_points: list[str] = Field(description="3 key points")
        difficulty: str = Field(description="beginner, intermediate, or advanced")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Explain the concept and return structured JSON."),
        ("user", "Explain {topic} for a {audience}. Return JSON matching the schema."),
    ])

    model = get_model()
    parser = JsonOutputParser(pydantic_object=Explanation)

    chain = prompt | model | parser

    result = chain.invoke({"topic": "neural networks", "audience": "college student"})
    print(f"Result: {result}\n")
    return result


def chain_with_fallback():
    """Chain with fallback model"""
    print("=== Chain with Fallback ===")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("user", "{question}"),
    ])

    primary_model = get_model()
    fallback_model = init_chat_model("openai/gpt-4o-mini")

    chain = prompt | primary_model.with_fallbacks([fallback_model]) | StrOutputParser()

    result = chain.invoke({"question": "What is the capital of France?"})
    print(f"Result: {result}\n")
    return result


def sequential_chains():
    """Multiple chains in sequence"""
    print("=== Sequential Chains ===")

    model = get_model()

    # Chain 1: Generate a topic
    topic_prompt = ChatPromptTemplate.from_template(
        "Suggest one interesting topic about {domain} for a blog post."
    )
    topic_chain = topic_prompt | model | StrOutputParser()

    # Chain 2: Create outline from topic
    outline_prompt = ChatPromptTemplate.from_template(
        "Create a 3-point outline for a blog post about: {topic}"
    )
    outline_chain = outline_prompt | model | StrOutputParser()

    # Chain 3: Write intro from outline
    intro_prompt = ChatPromptTemplate.from_template(
        "Write an engaging intro paragraph for a blog post with this outline:\n{outline}"
    )
    intro_chain = intro_prompt | model | StrOutputParser()

    # Combined chain
    full_chain = (
        {"topic": topic_chain}
        | {"outline": outline_chain, "topic": lambda x: x["topic"]}
        | {"intro": intro_chain, "outline": lambda x: x["outline"], "topic": lambda x: x["topic"]}
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