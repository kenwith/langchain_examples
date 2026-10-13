"""Example: Ollama LangGraph Subgraphs
===================================

This example demonstrates how to compose a parent LangGraph graph with a subgraph using Ollama models.

| Field        | Value                                                              |
|--------------|--------------------------------------------------------------------|
| Example      | Ollama LangGraph Subgraphs                                         |
| Description  | Parent graph with a subgraph for intermediate processing           |
| Models       | llama3.1 (Ollama)                                                  |
| Providers    | Ollama                                                             |
| Dependencies | langchain, langgraph, langchain-ollama                             |
"""

import os
from typing import TypedDict

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END


class ParentState(TypedDict):
    question: str
    intermediate: str
    final: str


class SubgraphState(TypedDict):
    question: str
    intermediate: str


model = init_chat_model(
    model=os.getenv("OLLAMA_MODEL", "llama3.1"),
    provider="ollama",
)


def generate_intermediate(state: SubgraphState) -> dict:
    """Generate an intermediate response for the question."""
    response = model.invoke(
        f"Provide a brief intermediate analysis for: {state['question']}"
    )
    return {"intermediate": response.content}


subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node("generate_intermediate", generate_intermediate)
subgraph_builder.add_edge(START, "generate_intermediate")
subgraph_builder.add_edge("generate_intermediate", END)
subgraph = subgraph_builder.compile()


def run_subgraph(state: ParentState) -> dict:
    """Invoke the subgraph to produce an intermediate result."""
    subgraph_result = subgraph.invoke({"question": state["question"]})
    return {"intermediate": subgraph_result["intermediate"]}


def generate_final(state: ParentState) -> dict:
    """Generate the final answer using the intermediate result."""
    response = model.invoke(
        f"Using this intermediate analysis: {state['intermediate']}\n"
        f"Answer the question: {state['question']}"
    )
    return {"final": response.content}


parent_builder = StateGraph(ParentState)
parent_builder.add_node("run_subgraph", run_subgraph)
parent_builder.add_node("generate_final", generate_final)
parent_builder.add_edge(START, "run_subgraph")
parent_builder.add_edge("run_subgraph", "generate_final")
parent_builder.add_edge("generate_final", END)
parent_graph = parent_builder.compile()


if __name__ == "__main__":
    result = parent_graph.invoke({"question": "What is LangGraph?"})
    print(result["final"])
