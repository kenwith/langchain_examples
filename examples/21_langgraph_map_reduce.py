"""
Example: Parallel map-reduce with LangGraph's Send API.

This example demonstrates how to process multiple documents in parallel using
LangGraph's Send API to fan out to individual tasks, then aggregate the results.
It uses a chat model to summarize each document, then combines the summaries.

To run, set your model provider API key (e.g., OPENAI_API_KEY) and optionally
change the model name in the code.

Requirements:
    - langgraph
    - langchain
    - a valid API key for your chosen model provider
"""

from typing import TypedDict, Annotated, List
import operator

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain.chains import init_chat_model

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Model identifier – change to your preferred provider/model.
# Format: "<provider>:<model_name>"
MODEL_NAME = "openai:gpt-4o"  # or "anthropic:claude-3-5-sonnet-20240620", etc.

# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------
class State(TypedDict):
    """State of the graph."""
    documents: List[str]          # input documents
    summaries: Annotated[List[str], operator.add]  # collected summaries (appended by reducer)

# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------
def map_documents(state: State):
    """
    Map node: fan out to a separate `summarize_one` task for each document.
    Returns a list of Send objects – one per document.
    """
    return [
        Send("summarize_one", {"doc": doc})
        for doc in state["documents"]
    ]


def summarize_one(state: dict):
    """
    Summarize a single document using the chat model.
    This node runs in parallel for each document.
    """
    # Initialize the model (provider-agnostic)
    model = init_chat_model(MODEL_NAME, temperature=0)

    # Generate a short summary
    doc = state["doc"]
    response = model.invoke(
        f"Please provide a concise one-sentence summary of the following document:\n\n{doc}"
    )
    summary = response.content.strip()

    # Return the summary wrapped in a list so the reducer appends it
    return {"summaries": [summary]}


def reduce_summaries(state: State):
    """
    Reduce node: combine all individual summaries into a single final summary.
    """
    summaries = state["summaries"]

    # Combine all summaries into one aggregated text
    combined = "\n".join(f"- {s}" for s in summaries)
    final_prompt = (
        "You have received the following summaries of separate documents.\n"
        "Please write a coherent overall summary that integrates the key points.\n\n"
        f"{combined}"
    )

    # Initialize the model again (or reuse a global instance)
    model = init_chat_model(MODEL_NAME, temperature=0)
    final_response = model.invoke(final_prompt)
    final_summary = final_response.content.strip()

    return {"final_summary": final_summary}


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------
def build_graph():
    """Construct the LangGraph state graph."""
    graph = StateGraph(State)

    # Add nodes
    graph.add_node("map_documents", map_documents)
    graph.add_node("summarize_one", summarize_one)
    graph.add_node("reduce_summaries", reduce_summaries)

    # Define edges
    graph.add_edge(START, "map_documents")
    graph.add_conditional_edges(
        "map_documents",
        lambda state: state,  # not used; we rely on the Send objects
        path_map=None,        # Send objects determine the next node
    )
    # Since map_documents returns Send objects, we need to specify how to route them.
    # The graph automatically routes each Send to the target node ("summarize_one").
    # We only need to add an edge from "summarize_one" to "reduce_summaries".
    graph.add_edge("summarize_one", "reduce_summaries")
    graph.add_edge("reduce_summaries", END)

    # Compile the graph
    return graph.compile()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Example documents
    documents = [
        "LangGraph is a library for building stateful, multi-actor applications with LLMs. "
        "It provides fine-grained control over both the flow and state of your application.",
        "The Send API in LangGraph allows you to dynamically fan out to multiple tasks, "
        "each with its own state, and then aggregate the results in a reducer.",
        "Map-reduce is a common pattern for processing large collections of data in parallel, "
        "often used in AI to summarize multiple documents at once.",
    ]

    # Initialize the graph
    app = build_graph()

    # Run the graph
    result = app.invoke({"documents": documents})

    # Print the final aggregated summary
    print("Final summary:\n", result["final_summary"])
