"""
Streaming Responses Example

Demonstrates: Token streaming, async streaming, streaming with tools, streaming structured output, concurrent streaming
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


# =============================================================================
# Concurrent Streaming with asyncio.gather
# =============================================================================

async def stream_single_prompt(prompt_text: str, model_name: str = None) -> AsyncGenerator[tuple[str, str], None]:
    """
    Stream a single prompt and yield (prompt_id, chunk) pairs.
    
    Args:
        prompt_text: The prompt to send to the model
        model_name: Optional model override
        
    Yields:
        Tuples of (prompt_identifier, token_chunk)
    """
    model = get_model() if model_name is None else init_chat_model(model_name)
    chain = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Be concise."),
        ("user", "{question}"),
    ]) | model | StrOutputParser()
    
    prompt_id = prompt_text[:30] + "..." if len(prompt_text) > 30 else prompt_text
    async for chunk in chain.astream({"question": prompt_text}):
        yield (prompt_id, chunk)


async def concurrent_streaming_gather():
    """
    Demonstrate concurrent streaming of multiple prompts using asyncio.gather.
    
    This pattern allows streaming responses from multiple model calls simultaneously,
    interleaving tokens as they arrive from each stream.
    """
    print("=== Concurrent Streaming with asyncio.gather ===")
    
    prompts = [
        "Write a haiku about Python",
        "Write a haiku about JavaScript", 
        "Write a haiku about Rust",
        "Write a haiku about Go",
    ]
    
    # Create async generators for each prompt
    async def collect_stream(prompt: str) -> list[tuple[str, str]]:
        """Collect all chunks from a single stream into a list."""
        chunks = []
        async for prompt_id, chunk in stream_single_prompt(prompt):
            chunks.append((prompt_id, chunk))
        return chunks
    
    # Run all streams concurrently and collect results
    print("Starting concurrent streams...\n")
    results = await asyncio.gather(*[collect_stream(p) for p in prompts])
    
    # Print results grouped by prompt
    for prompt_chunks in results:
        prompt_id = prompt_chunks[0][0] if prompt_chunks else "unknown"
        print(f"--- {prompt_id} ---")
        for _, chunk in prompt_chunks:
            print(chunk, end="", flush=True)
        print("\n")


async def concurrent_streaming_interleaved():
    """
    Demonstrate truly interleaved concurrent streaming.
    
    This version yields tokens as they arrive from any stream,
    showing real-time interleaving of multiple model responses.
    """
    print("=== Interleaved Concurrent Streaming ===")
    
    prompts = [
        "Count from 1 to 3",
        "List 3 colors",
        "List 3 animals",
    ]
    
    # Create generators for each prompt
    generators = [stream_single_prompt(p) for p in prompts]
    
    # Use asyncio.as_completed style pattern for true interleaving
    # We'll create tasks that yield chunks as they arrive
    async def stream_with_label(gen: AsyncGenerator[tuple[str, str], None], label: str):
        """Wrap generator to add a label for identification."""
        async for prompt_id, chunk in gen:
            yield (label, chunk)
    
    labeled_generators = [
        stream_with_label(gen, f"Task-{i}") 
        for i, gen in enumerate(generators)
    ]
    
    # Create a merged async generator that yields from all streams as they produce
    async def merged_stream():
        # Create tasks for each generator's iteration
        pending = {
            asyncio.create_task(gen.__anext__()): gen 
            for gen in labeled_generators
        }
        
        while pending:
            done, pending = await asyncio.wait(
                pending, 
                return_when=asyncio.FIRST_COMPLETED
            )
            
            for task in done:
                gen = pending.pop(task)
                try:
                    label, chunk = task.result()
                    yield (label, chunk)
                    # Schedule next chunk from this generator
                    new_task = asyncio.create_task(gen.__anext__())
                    pending[new_task] = gen
                except StopAsyncIteration:
                    # This generator is exhausted
                    pass
    
    print("Streaming interleaved responses:\n")
    async for label, chunk in merged_stream():
        print(f"[{label}] {chunk}", end="", flush=True)
    print("\n")


async def concurrent_streaming_with_semaphore():
    """
    Demonstrate concurrent streaming with a semaphore for rate limiting.
    
    Useful when you want to limit the number of concurrent model calls
    to avoid rate limits or resource exhaustion.
    """
    print("=== Concurrent Streaming with Semaphore (Rate Limited) ===")
    
    prompts = [f"Write a one-sentence fact about topic {i}" for i in range(6)]
    semaphore = asyncio.Semaphore(2)  # Max 2 concurrent streams
    
    async def limited_stream(prompt: str) -> list[tuple[str, str]]:
        async with semaphore:
            chunks = []
            async for prompt_id, chunk in stream_single_prompt(prompt):
                chunks.append((prompt_id, chunk))
            return chunks
    
    print("Running 6 prompts with max 2 concurrent...\n")
    results = await asyncio.gather(*[limited_stream(p) for p in prompts])
    
    for prompt_chunks in results:
        prompt_id = prompt_chunks[0][0] if prompt_chunks else "unknown"
        print(f"--- {prompt_id} ---")
        for _, chunk in prompt_chunks:
            print(chunk, end="", flush=True)
        print("\n")


async def concurrent_structured_streaming():
    """
    Demonstrate concurrent streaming with structured output parsing.
    
    Shows how to run multiple structured output extractions concurrently.
    """
    print("=== Concurrent Structured Output Streaming ===")
    
    from langchain_core.output_parsers import JsonOutputParser
    
    class Fact(BaseModel):
        topic: str = Field(description="The topic")
        fact: str = Field(description="A fun fact")
        confidence: float = Field(description="Confidence 0-1")
    
    parser = JsonOutputParser(pydantic_object=Fact)
    model = get_model()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Return a fun fact as JSON. {format_instructions}"),
        ("user", "Give me a fun fact about {topic}"),
    ])
    
    chain = prompt | model | parser
    
    topics = ["octopus", "honeybee", "banana", "lightning"]
    
    async def extract_fact(topic: str) -> Fact:
        result = await chain.ainvoke({
            "topic": topic,
            "format_instructions": parser.get_format_instructions()
        })
        return result
    
    print("Extracting structured facts concurrently...\n")
    facts = await asyncio.gather(*[extract_fact(t) for t in topics])
    
    for fact in facts:
        print(f"  {fact['topic']}: {fact['fact']} (confidence: {fact['confidence']})")
    print()


if __name__ == "__main__":
    basic_streaming()
    asyncio.run(async_streaming())
    streaming_with_tools()
    langgraph_stream_modes()
    streaming_structured()
    asyncio.run(demo_custom_generator())
    asyncio.run(concurrent_streaming_gather())
    asyncio.run(concurrent_streaming_interleaved())
    asyncio.run(concurrent_streaming_with_semaphore())
    asyncio.run(concurrent_structured_streaming())
    print("\nAll streaming examples completed!")
