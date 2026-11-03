"""LangGraph workflow example.

This module demonstrates a simple LangGraph workflow with a state schema defined
as a TypedDict. The state is a dictionary that is passed between nodes. Each node
returns a partial dictionary of updates, which LangGraph merges into the current
state, allowing subsequent nodes to access the accumulated data.

State schema:
- query: The user's input question.
- answer: The final generated answer.
- steps: A list of strings recording the intermediate processing steps.

Nodes update state by returning a dictionary with only the keys they modify.
For example, the retrieve node adds retrieved documents to the state, while the
generate node uses the accumulated state to produce the final answer.
"""

from typing import TypedDict, List

from langgraph.graph import StateGraph, END


class WorkflowState(TypedDict):
    """State schema for the workflow."""
    query: str
    answer: str
    steps: List[str]


def retrieve_node(state: WorkflowState) -> dict:
    """Simulate document retrieval and update state with a step."""
    return {"steps": state["steps"] + [f"Retrieved documents for: {state['query']}"]}


def generate_node(state: WorkflowState) -> dict:
    """Generate an answer based on the current state."""
    return {
        "answer": f"Answer to '{state['query']}'",
        "steps": state["steps"] + ["Generated answer"],
    }


def build_graph():
    """Build and compile the LangGraph workflow."""
    graph = StateGraph(WorkflowState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


def main():
    """Run the example workflow."""
    app = build_graph()
    result = app.invoke({"query": "What is LangGraph?", "steps": []})
    print(result["answer"])
    print("Steps:")
    for step in result["steps"]:
        print(f"- {step}")


if __name__ == "__main__":
    main()
