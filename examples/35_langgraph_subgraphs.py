"""Example 35: LangGraph Subgraphs

This example demonstrates how to compose a child graph inside a parent graph
to create reusable workflow components. The child graph is a simple summarizer
that takes a piece of text and produces a summary. The parent graph splits a
larger document into chunks, uses the child graph to summarize each chunk, and
then combines the summaries into a final overview.

This example is provider-agnostic; it uses `init_chat_model` to create a chat
model based on environment variables (e.g., `CHAT_MODEL` and `CHAT_MODEL_PROVIDER`).
No API keys are hardcoded; use environment variables or a secrets manager.
"""

import os
from typing import List, TypedDict

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END


# ---------------------------------------------------------------------------
# Child graph: Summarize a single piece of text
# ---------------------------------------------------------------------------

class ChildState(TypedDict):
    """State for the child summarization graph."""
    text: str
    summary: str


def summarize_text(state: ChildState) -> dict:
    """Node that uses the LLM to summarize the input text."""
    llm = init_chat_model(
        os.getenv("CHAT_MODEL", "gpt-4o"),
        model_provider=os.getenv("CHAT_MODEL_PROVIDER", "openai"),
    )
    prompt = f"Please summarize the following text in a few sentences:\n\n{state['text']}"
    response = llm.invoke(prompt)
    return {"summary": response.content}


# Build the child graph
child_graph = StateGraph(ChildState)
child_graph.add_node("summarize", summarize_text)
child_graph.add_edge(START, "summarize")
child_graph.add_edge("summarize", END)
child_compiled = child_graph.compile()


# ---------------------------------------------------------------------------
# Parent graph: Split a document, summarize each chunk, combine summaries
# ---------------------------------------------------------------------------

class ParentState(TypedDict):
    """State for the parent graph that orchestrates the child graph."""
    text: str
    chunks: List[str]
    summaries: List[str]
    final_summary: str


def split_text(state: ParentState) -> dict:
    """Split the input text into smaller chunks (by paragraphs)."""
    paragraphs = [p.strip() for p in state["text"].split("\n\n") if p.strip()]
    return {"chunks": paragraphs}


def summarize_chunks(state: ParentState) -> dict:
    """Summarize each chunk using the child graph."""
    summaries = []
    for chunk in state["chunks"]:
        result = child_compiled.invoke({"text": chunk})
        summaries.append(result["summary"])
    return {"summaries": summaries}


def combine_summaries(state: ParentState) -> dict:
    """Combine all chunk summaries into a final overview."""
    llm = init_chat_model(
        os.getenv("CHAT_MODEL", "gpt-4o"),
        model_provider=os.getenv("CHAT_MODEL_PROVIDER", "openai"),
    )
    joined = "\n".join(f"- {s}" for s in state["summaries"])
    prompt = (
        "You are given several summaries from different sections of a document. "
        "Write a coherent final summary that ties them together:\n\n"
        f"{joined}"
    )
    response = llm.invoke(prompt)
    return {"final_summary": response.content}


# Build the parent graph
parent_graph = StateGraph(ParentState)
parent_graph.add_node("split", split_text)
parent_graph.add_node("summarize", summarize_chunks)
parent_graph.add_node("combine", combine_summaries)

parent_graph.add_edge(START, "split")
parent_graph.add_edge("split", "summarize")
parent_graph.add_edge("summarize", "combine")
parent_graph.add_edge("combine", END)

parent_compiled = parent_graph.compile()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_text = """\
LangGraph is a library for building stateful, multi-step applications with LLMs.
It provides a flexible graph-based framework where you can define nodes and edges
to control the flow of data.

One of the key features is the ability to compose subgraphs. This allows you to
reuse complex workflows as building blocks within larger graphs.

The library integrates well with LangChain, making it easy to combine language
models, tools, and other components.
"""

    print("Running parent graph with sample text...\n")
    result = parent_compiled.invoke({"text": sample_text})
    print("Final Summary:")
    print(result["final_summary"])
