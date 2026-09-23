"""Use `interrupt` and `Command` to pause a graph for human approval.

This example builds a LangGraph that generates a draft response with a
provider-agnostic chat model, then pauses to ask a human for approval before
returning the final result. The graph uses `interrupt` to pause and
`Command(resume=...)` to continue.

Set the `MODEL_NAME` environment variable to choose a model, e.g.
`openai:gpt-4o-mini` or `anthropic:claude-3-5-sonnet-latest`. You will also
need the corresponding API key set in your environment.
"""

# ---------------------------------------------------------------------------
# Example: 34_langgraph_human_in_the_loop.py
# Description: Use `interrupt` and `Command` to pause a graph for human approval.
# ---------------------------------------------------------------------------

import os
from typing import TypedDict

from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


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
    """Pause the graph and ask a human to approve the draft."""
    response = interrupt(
        {
            "question": "Do you approve this draft?",
            "draft": state["draft"],
        }
    )

    # The response comes from Command(resume=...). It can be a dict with an
    # "approved" key, or a simple boolean. Default to False for safety.
    if isinstance(response, dict):
        approved = bool(response.get("approved", False))
    else:
        approved = bool(response)

    return {"approved": approved}


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

    return graph.compile()


def main():
    """Run the human-in-the-loop example."""
    graph = build_graph()
    config = {"configurable": {"thread_id": "demo-thread"}}

    # Start the graph. It will pause in `human_review` with an interrupt.
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

    # Resume the graph with human approval.
    result = graph.invoke(
        Command(resume={"approved": True}),
        config,
    )
    print("\nFinal result after approval:")
    print(result["result"])


if __name__ == "__main__":
    main()
