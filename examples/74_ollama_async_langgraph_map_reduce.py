"""Async map-reduce workflow with LangGraph and Ollama.

This example demonstrates an asynchronous map-reduce workflow using LangGraph
with Ollama as the model provider. The map step processes multiple topics in
parallel, and the reduce step combines the resulting summaries into a single
final summary. The model is created with the provider-agnostic
``init_chat_model`` helper.
"""

import asyncio
import os
import operator
from typing import Annotated, List, TypedDict

from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph


class MapReduceState(TypedDict):
    """State for the map-reduce workflow."""
    topics: List[str]
    summaries: Annotated[List[str], operator.add]
    final_summary: str


def get_model():
    """Create a chat model using Ollama via init_chat_model."""
    return init_chat_model(
        model=os.getenv("OLLAMA_MODEL", "llama3.1"),
        provider="ollama",
        temperature=0,
    )


async def map_node(state: MapReduceState) -> dict:
    """Generate a summary for each topic in parallel."""
    model = get_model()
    tasks = [
        model.ainvoke(f"Write a short summary of {topic}.")
        for topic in state["topics"]
    ]
    responses = await asyncio.gather(*tasks)
    summaries = [response.content for response in responses]
    return {"summaries": summaries}


async def reduce_node(state: MapReduceState) -> dict:
    """Combine all summaries into one final summary."""
    model = get_model()
    combined = "\n".join(state["summaries"])
    prompt = f"Combine these summaries into one concise summary:\n{combined}"
    response = await model.ainvoke(prompt)
    return {"final_summary": response.content}


# Build the graph.
graph = StateGraph(MapReduceState)
graph.add_node("map", map_node)
graph.add_node("reduce", reduce_node)
graph.add_edge(START, "map")
graph.add_edge("map", "reduce")
graph.add_edge("reduce", END)
app = graph.compile()


async def main() -> None:
    """Run the map-reduce workflow with sample topics."""
    topics = ["LangChain", "LangGraph", "Ollama"]
    result = await app.ainvoke({"topics": topics})
    print(result["final_summary"])


if __name__ == "__main__":
    asyncio.run(main())
