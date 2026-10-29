"""Use `interrupt_before` and `Command` to pause a graph for human approval.

This example builds a LangGraph that generates a draft response with a
provider-agnostic chat model, then pauses for human approval before
returning the final result. The graph uses an explicit `interrupt_before`
checkpoint to pause before a dedicated `approval` node, and resumes with
`Command(resume=...)`. After the approval node, conditional edges route
to either an approved finalization path or a rejection path, making the
human decision visible in the graph structure.

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


def approval(state: State) -> dict:
    """Validate the human decision after the interrupt checkpoint.

    The approval decision is merged into the state by `Command(resume=...)`.
    This node is the dedicated approval checkpoint where additional review
    logic can be applied before routing to the appropriate final output.
    """
    if "approved" not in state:
        raise ValueError("Approval decision missing from state.")
    return {}


def route_after_approval(state: State) -> str:
    """Route to finalize or reject based on the human approval decision."""
    if state.get("approved"):
        return "finalize"
    return "reject"


def finalize(state: State) -> dict:
    """Return the approved draft as the final result."""
    return {"result": state["draft"]}


def reject(state: State) -> dict:
    """Return a rejection message when the human declines the draft."""
    return {"result": "Request rejected by human."}


def build_graph():
    """Build and compile the human-in-the-loop graph."""
    graph = StateGraph(State)
    graph.add_node("generate_draft", generate_draft)
    graph.add_node("approval", approval)
    graph.add_node("finalize", finalize)
    graph.add_node("reject", reject)

    graph.add_edge(START, "generate_draft")
    graph.add_edge("generate_draft", "approval")
    graph.add_conditional_edges(
        "approval",
        route_after_approval,
        {"finalize": "finalize", "reject": "reject"},
    )
    graph.add_edge("finalize", END)
    graph.add_edge("reject", END)

    # Compile the graph with an explicit interrupt before the approval node.
    # This creates a checkpoint: the first invocation executes all nodes up
    # to but not including `approval`, then returns with an `__interrupt__`
    # payload instead of a final result. The application can then present
    # the draft to a human and resume the graph with their decision.
    return graph.compile(interrupt_before=["approval"])


def resume_graph(graph, config, approved: bool):
    """Resume the graph with the human approval decision."""
    return graph.invoke(Command(resume={"approved": approved}), config)


def main():
    """Run the human-in-the-loop example."""
    graph = build_graph()
    config = {"configurable": {"thread_id": "demo-thread"}}

    # Invoking the graph for the first time runs `generate_draft` and then
    # pauses before `approval`. The returned state includes an `__interrupt__`
    # key containing the interrupt payload. Execution is suspended at the
    # `approval` checkpoint and resumes only when `Command(resume=...)` is
    # passed to invoke with the same thread config.
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

    # Resume with `Command(resume={"approved": ...})`. The command value is
    # merged into the state before the interrupted `approval` node is
    # executed, and the conditional edges then follow the visible
    # approve/reject routing.
    result = resume_graph(graph, config, approved=True)
    print("\nFinal result after approval:")
    print(result["result"])


if __name__ == "__main__":
    main()
