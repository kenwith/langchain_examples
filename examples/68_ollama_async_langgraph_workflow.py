"""Example: 68 - Ollama Async LangGraph Workflow

Description: Demonstrates a local async graph using Ollama, async execution,
and LangGraph parallel nodes.
"""

from __future__ import annotations

import asyncio
import os
from typing import TypedDict

from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph


class WorkflowState(TypedDict):
    topic: str
    result_one: str
    result_two: str
    combined: str


async def generate_idea_one(state: WorkflowState) -> dict:
    """Generate a first idea using the local Ollama model."""
    model = init_chat_model(
        model=os.getenv("OLLAMA_MODEL", "llama3.1"),
        provider="ollama",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    prompt = f"Give a one-sentence idea about {state['topic']}."
    response = await model.ainvoke(prompt)
    return {"result_one": response.content}


async def generate_idea_two(state: WorkflowState) -> dict:
    """Generate a second idea using the local Ollama model."""
    model = init_chat_model(
        model=os.getenv("OLLAMA_MODEL", "llama3.1"),
        provider="ollama",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    prompt = f"Give a different one-sentence idea about {state['topic']}."
    response = await model.ainvoke(prompt)
    return {"result_two": response.content}


async def combine_ideas(state: WorkflowState) -> dict:
    """Combine the two parallel results into a single summary."""
    model = init_chat_model(
        model=os.getenv("OLLAMA_MODEL", "llama3.1"),
        provider="ollama",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    prompt = (
        f"Combine these two ideas into one actionable concept:\n"
        f"1. {state['result_one']}\n2. {state['result_two']}"
    )
    response = await model.ainvoke(prompt)
    return {"combined": response.content}


async def run_workflow(topic: str) -> None:
    """Build and execute the async LangGraph workflow."""
    workflow = StateGraph(WorkflowState)

    workflow.add_node("generate_idea_one", generate_idea_one)
    workflow.add_node("generate_idea_two", generate_idea_two)
    workflow.add_node("combine_ideas", combine_ideas)

    workflow.add_edge(START, "generate_idea_one")
    workflow.add_edge(START, "generate_idea_two")
    workflow.add_edge("generate_idea_one", "combine_ideas")
    workflow.add_edge("generate_idea_two", "combine_ideas")
    workflow.add_edge("combine_ideas", END)

    app = workflow.compile()

    result = await app.ainvoke({"topic": topic})
    print(f"Topic: {topic}")
    print(f"Idea 1: {result['result_one']}")
    print(f"Idea 2: {result['result_two']}")
    print(f"Combined: {result['combined']}")


if __name__ == "__main__":
    asyncio.run(run_workflow("sustainable urban mobility"))
