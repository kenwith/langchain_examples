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

The workflow includes a conditional edge after the generate node. If the original
query contains the word "complex", the answer is routed to a refine node for
additional processing; otherwise, the workflow terminates. This branching
demonstrates how LangGraph can dynamically decide the next step based on state.
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


def refine_node(state: WorkflowState) -> dict:
    """Refine the answer by appending a marker and recording the step."""
    return {
        "answer": state["answer"] + " (refined)",
        "steps": state["steps"] + ["Refined answer"],
    }


def route_after_generate(state: WorkflowState) -> str:
    """Decide whether to refine the answer or end the workflow."""
    # If the query contains "complex", route to the refine node.
    if "complex" in state["query"].lower():
        return "refine"
    # Otherwise, end the workflow.
    return "end"


def build_graph():
    """Build and compile the LangGraph workflow."""
    graph = StateGraph(WorkflowState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("refine", refine_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")

    # Add a conditional edge from generate to either refine or END.
    # The route_after_generate function inspects the state and returns a key.
    graph.add_conditional_edges(
        "generate",
        route_after_generate,
        {
            "refine": "refine",
            "end": END,
        },
    )

    graph.add_edge("refine", END)

    return graph.compile()


def main():
    """Run the example workflow."""
    app = build_graph()
    result = app.invoke({"query": "What is LangGraph?", "steps": []})
    print(result["answer"])
    print("Steps:")
    for step in result["steps"]:
        print(f"- {step}")

    # Demonstrate the conditional path with a query containing "complex".
    result_complex = app.invoke({"query": "Explain complex workflows", "steps": []})
    print("\nSecond run (complex query):")
    print(result_complex["answer"])
    print("Steps:")
    for step in result_complex["steps"]:
        print(f"- {step}")


if __name__ == "__main__":
    main()
