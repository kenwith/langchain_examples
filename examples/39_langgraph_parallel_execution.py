"""
Example: LangGraph Parallel Execution
=====================================

This example demonstrates how to run multiple independent nodes in parallel
within a LangGraph state graph and combine their results into a final output.

| Name | Description |
|------|-------------|
| 39_langgraph_parallel_execution.py | Run multiple branches concurrently and merge their outputs |

The graph first generates a topic, then two independent nodes write a short
story and a poem about that topic, and finally a combine node merges both
pieces into a single response.
"""

from typing import TypedDict

from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph

# Use a provider-agnostic model; configure via environment variables.
# Example: export MODEL="gpt-4o-mini" (or "anthropic:claude-3-5-sonnet-20240620")
model = init_chat_model(
    model=os.getenv("MODEL", "gpt-4o-mini"),
    temperature=0.7,
)


class GraphState(TypedDict):
    """State of the LangGraph graph."""
    topic: str
    story: str
    poem: str
    combined_output: str


def generate_topic(state: GraphState) -> dict:
    """Generate a creative topic if none is provided."""
    if "topic" in state and state["topic"]:
        return {"topic": state["topic"]}
    response = model.invoke(
        "Suggest a single, interesting topic for a short story and a poem. "
        "Return only the topic name, no extra text."
    )
    return {"topic": response.content.strip()}


def write_story(state: GraphState) -> dict:
    """Write a short story about the topic."""
    prompt = f"Write a short story (2-3 sentences) about: {state['topic']}"
    response = model.invoke(prompt)
    return {"story": response.content.strip()}


def write_poem(state: GraphState) -> dict:
    """Write a short poem about the topic."""
    prompt = f"Write a short poem (2-3 lines) about: {state['topic']}"
    response = model.invoke(prompt)
    return {"poem": response.content.strip()}


def combine_results(state: GraphState) -> dict:
    """Combine the story and poem into a final output."""
    combined = (
        f"Topic: {state['topic']}\n\n"
        f"Story:\n{state['story']}\n\n"
        f"Poem:\n{state['poem']}"
    )
    return {"combined_output": combined}


def build_graph():
    """Build the LangGraph graph."""
    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("generate_topic", generate_topic)
    graph.add_node("write_story", write_story)
    graph.add_node("write_poem", write_poem)
    graph.add_node("combine_results", combine_results)

    # Define edges
    graph.add_edge(START, "generate_topic")
    graph.add_edge("generate_topic", "write_story")
    graph.add_edge("generate_topic", "write_poem")
    graph.add_edge("write_story", "combine_results")
    graph.add_edge("write_poem", "combine_results")
    graph.add_edge("combine_results", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()
    result = app.invoke({"topic": ""})  # empty topic triggers generation
    print(result["combined_output"])
