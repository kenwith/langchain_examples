"""Use `interrupt_before` and `Command` to pause a graph for human approval.

This example builds a LangGraph that generates a draft response with a
provider-agnostic chat model, then pauses for human approval before
returning the final result. The graph uses an explicit `interrupt_before`
checkpoint to pause before the `human_review` node, and resumes with
`Command(resume=...)`.

Set the `MODEL_NAME` environment variable to choose a model, e.g.
`openai:gpt-4o-mini` or `anthropic:claude-3-5-sonnet-latest`. You will also
need the corresponding API key set in your environment.
"""

# ---------------------------------------------------------------------------
# Example: 34_langgraph_human_in_the_loop.py
# Description: Use `interrupt_before` and `Command` to pause a graph for human approval.
# ---------------------------------------------------------------------------

import os
from typing import TypedDict

from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command


class State(TypedDict):
    query: str
    draft: str
    approved: bool
    result: str


def generate_draft(state: State) -> dict:
    """Generate a draft response using a provider-agnostic chat model."""
    model = init_chat_model(os.getenv("MODEL_NAME", "openai:gpt-4o-mini"))
    response = model.invoke(state["query"])
    return {"draft": response.content}


def human_review(state: State) -> dict:
    """Process the human decision after the interrupt checkpoint.

    The approval decision is merged into the state by `Command(resume=...)`.
    This node runs after the graph resumes and exists primarily as the
    explicit checkpoint target.
    """
    return {}


def finalize(state: State) -> dict:
    """Return the final result based on human approval."""
    if state["approved"]:
        return {"result": state["draft"]}
    return {"result": "Request rejected by human."}


def build_graph():
    """Build and compile the human-in-the-loop graph."""
    graph = StateGraph(State)
    graph.add_node("generate_draft", generate_draft)
    graph.add_node("human_review", human_review)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "generate_draft")
    graph.add_edge("generate_draft", "human_review")
    graph.add_edge("human_review", "finalize")
    graph.add_edge("finalize", END)

    # Explicit checkpoint: pause before the human_review node.
    return graph.compile(interrupt_before=["human_review"])


def resume_graph(graph, config, approved: bool):
    """Resume the graph with the human approval decision."""
    return graph.invoke(Command(resume={"approved": approved}), config)


def main():
    """Run the human-in-the-loop example."""
    graph = build_graph()
    config = {"configurable": {"thread_id": "demo-thread"}}

    # Start the graph. It will pause before `human_review` due to the
    # explicit `interrupt_before` checkpoint.
    result = graph.invoke(
        {"query": "Explain LangGraph in one sentence."},
        config,
    )

    if "__interrupt__" in result:
        interrupt_value = result["__interrupt__"][0].value
        print("Human approval requested:")
        print(interrupt_value)
    else:
        print("Unexpected: graph did not interrupt.")
        return

    # Resume the graph with human approval using the dedicated resume function.
    result = resume_graph(graph, config, approved=True)
    print("\nFinal result after approval:")
    print(result["result"])


if __name__ == "__main__":
    main()
