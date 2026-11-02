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

The workflow is exposed through the `build_graph` function, which returns a
compiled LangGraph graph. A `run` helper is also provided for convenient
execution and testing.
"""

import os
from typing import TypedDict

from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph


class GraphState(TypedDict):
    """State of the LangGraph graph."""
    topic: str
    story: str
    poem: str
    combined_output: str


def generate_topic(state: GraphState) -> dict:
    """Generate a creative topic if none is provided."""
    if state.get("topic"):
        return {"topic": state["topic"]}
    # Initialize the model lazily inside the node to avoid side effects on import
    model = init_chat_model(
        model=os.getenv("MODEL", "gpt-4o-mini"),
        temperature=0.7,
    )
    response = model.invoke(
        "Suggest a single, interesting topic for a short story and a poem. "
        "Return only the topic name, no extra text."
    )
    return {"topic": response.content.strip()}


def write_story(state: GraphState) -> dict:
    """Write a short story about the topic."""
    model = init_chat_model(
        model=os.getenv("MODEL", "gpt-4o-mini"),
        temperature=0.7,
    )
    prompt = f"Write a short story (2-3 sentences) about: {state['topic']}"
    response = model.invoke(prompt)
    return {"story": response.content.strip()}


def write_poem(state: GraphState) -> dict:
    """Write a short poem about the topic."""
    model = init_chat_model(
        model=os.getenv("MODEL", "gpt-4o-mini"),
        temperature=0.7,
    )
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


def build_graph(model=None, temperature: float = 0.7):
    """Build the LangGraph graph.

    Args:
        model: An optional model instance. If None, a model is created from
            the MODEL environment variable (default "gpt-4o-mini") using the
            provided temperature.
        temperature: The temperature to use when creating a model. Ignored if
            `model` is provided.

    Returns:
        A compiled LangGraph graph.
    """
    # Create the model once if not injected, so all nodes share the same instance.
    if model is None:
        model = init_chat_model(
            model=os.getenv("MODEL", "gpt-4o-mini"),
            temperature=temperature,
        )

    def _generate_topic(state: GraphState) -> dict:
        """Generate a topic using the shared model."""
        if state.get("topic"):
            return {"topic": state["topic"]}
        response = model.invoke(
            "Suggest a single, interesting topic for a short story and a poem. "
            "Return only the topic name, no extra text."
        )
        return {"topic": response.content.strip()}

    def _write_story(state: GraphState) -> dict:
        """Write a short story using the shared model."""
        prompt = f"Write a short story (2-3 sentences) about: {state['topic']}"
        response = model.invoke(prompt)
        return {"story": response.content.strip()}

    def _write_poem(state: GraphState) -> dict:
        """Write a short poem using the shared model."""
        prompt = f"Write a short poem (2-3 lines) about: {state['topic']}"
        response = model.invoke(prompt)
        return {"poem": response.content.strip()}

    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("generate_topic", _generate_topic)
    graph.add_node("write_story", _write_story)
    graph.add_node("write_poem", _write_poem)
    graph.add_node("combine_results", combine_results)

    # Define edges
    graph.add_edge(START, "generate_topic")
    graph.add_edge("generate_topic", "write_story")
    graph.add_edge("generate_topic", "write_poem")
    graph.add_edge("write_story", "combine_results")
    graph.add_edge("write_poem", "combine_results")
    graph.add_edge("combine_results", END)

    return graph.compile()


def run(topic: str = "") -> str:
    """Run the graph with the given topic and return the combined output.

    Args:
        topic: The topic to use. If empty, the graph will generate one.

    Returns:
        The combined story and poem as a string.
    """
    app = build_graph()
    result = app.invoke({"topic": topic})
    return result["combined_output"]


def main() -> None:
    """Execute the example and print the combined output."""
    print(run(""))


if __name__ == "__main__":
    main()
