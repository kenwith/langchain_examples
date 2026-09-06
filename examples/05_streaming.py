"""
Streaming Responses Example

Demonstrates: Token streaming, async streaming, streaming with tools, streaming structured output
Provider-agnostic using init_chat_model
"""
import os
import asyncio
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing import Annotated, AsyncGenerator

load_dotenv()


def get_model():
    model_name = os.getenv("LANGCHAIN_MODEL", "openai/gpt-4o-mini")
    return init_chat_model(model_name)


# =============================================================================
# Basic Token Streaming
# =============================================================================

def basic_streaming():
    """Stream tokens from model"""
    print("=== Basic Token Streaming ===")

    model = get_model()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("user", "{question}"),
    ])
    chain = prompt | model | StrOutputParser()

    print("Streaming response:")
    for chunk in chain.stream({"question": "Write a short poem about streaming data"}):
        print(chunk, end="", flush=True)
    print("\n")


# =============================================================================
# Async Streaming
# =============================================================================

async def async_streaming():
    """Async token streaming"""
    print("=== Async Token Streaming ===")

    model = get_model()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("user", "{question}"),
    ])
    chain = prompt | model | StrOutputParser()

    print("Async streaming response:")
    async for chunk in chain.astream({"question": "Explain async streaming in 2 sentences"}):
        print(chunk, end="", flush=True)
    print("\n")


# =============================================================================
# Streaming with Tools (LangGraph)
# =============================================================================

def streaming_with_tools():
    """Stream from a LangGraph agent with tools"""
    print("=== Streaming with Tools ===")

    from langchain_core.tools import tool
    from langgraph.prebuilt import create_react_agent

    @tool
    def get_fact(topic: str) -> str:
        """Get a fun fact about a topic."""
        facts = {
            "python": "Python was named after Monty Python, not the snake.",
            "space": "A day on Venus is longer than a year on Venus.",
            "ocean": "We know more about the moon than the ocean floor.",
        }
        return facts.get(topic.lower(), f"No fact available for {topic}")

    model = get_model()
    agent = create_react_agent(model, [get_fact])

    print("Streaming agent response:")
    for chunk in agent.stream(
        {"messages": [HumanMessage(content="Give me a fun fact about Python")]},
        stream_mode="values"
    ):
        # chunk is the full state at each step
        last_msg = chunk["messages"][-1]
        if hasattr(last_msg, "content") and last_msg.content:
            print(f"\n[{last_msg.type}]: {last_msg.content}")
    print()


# =============================================================================
# Streaming Modes in LangGraph
# =============================================================================

def langgraph_stream_modes():
    """Different LangGraph stream modes"""
    print("=== LangGraph Stream Modes ===")

    model = get_model()

    class StreamState(BaseModel):
        messages: Annotated[list, add_messages] = Field(default_factory=list)
        step: str = ""

    def node_a(state: StreamState):
        result = model.invoke(state.messages + [HumanMessage(content="Step A: Say hello briefly")])
        return {"messages": [result], "step": "A"}

    def node_b(state: StreamState):
        result = model.invoke(state.messages + [HumanMessage(content="Step B: Say goodbye briefly")])
        return {"messages": [result], "step": "B"}

    workflow = StateGraph(StreamState)
    workflow.add_node("a", node_a)
    workflow.add_node("b", node_b)
    workflow.add_edge(START, "a")
    workflow.add_edge("a", "b")
    workflow.add_edge("b", END)

    app = workflow.compile()

    print("--- stream_mode='values' (full state each step) ---")
    for chunk in app.stream(
        {"messages": [HumanMessage(content="Start")]},
        stream_mode="values"
    ):
        print(f"Step: {chunk['step']}, Last msg: {chunk['messages'][-1].content[:50]}...")

    print("\n--- stream_mode='updates' (only changes) ---")
    for chunk in app.stream(
        {"messages": [HumanMessage(content="Start")]},
        stream_mode="updates"
    ):
        for node, update in chunk.items():
            print(f"Node '{node}' updated: {list(update.keys())}")

    print("\n--- stream_mode='messages' (token streaming) ---")
    for chunk in app.stream(
        {"messages": [HumanMessage(content="Start")]},
        stream_mode="messages"
    ):
        msg, metadata = chunk
        if hasattr(msg, "content") and msg.content:
            print(msg.content, end="", flush=True)
    print("\n")


# =============================================================================
# Streaming Structured Output
# =============================================================================

def streaming_structured():
    """Stream structured output (JSON)"""
    print("=== Streaming Structured Output ===")

    from langchain_core.output_parsers import JsonOutputParser

    class Poem(BaseModel):
        title: str = Field(description="Poem title")
        lines: list[str] = Field(description="Poem lines")
        theme: str = Field(description="Poem theme")

    model = get_model()
    parser = JsonOutputParser(pydantic_object=Poem)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Write a poem and return JSON. {format_instructions}"),
        ("user", "Write a poem about {topic}"),
    ])

    chain = prompt | model | parser

    # Note: JSON streaming requires special handling
    # This demonstrates the concept - full JSON comes at once
    result = chain.invoke({
        "topic": "streaming data",
        "format_instructions": parser.get_format_instructions()
    })
    print(f"Structured result: {result}")


# =============================================================================
# Custom Async Generator
# =============================================================================

async def custom_stream_generator() -> AsyncGenerator[str, None]:
    """Custom async generator for streaming"""
    model = get_model()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Count from 1 to 5, one number per line."),
        ("user", "Count"),
    ])
    chain = prompt | model | StrOutputParser()

    async for chunk in chain.astream({}):
        yield chunk


async def demo_custom_generator():
    print("=== Custom Async Generator ===")
    async for chunk in custom_stream_generator():
        print(f"Got: {chunk.strip()}")


if __name__ == "__main__":
    basic_streaming()
    asyncio.run(async_streaming())
    streaming_with_tools()
    langgraph_stream_modes()
    streaming_structured()
    asyncio.run(demo_custom_generator())
    print("\nAll streaming examples completed!")