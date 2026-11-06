"""LangGraph workflow example.

This module demonstrates a LangGraph workflow with a state schema defined as a
TypedDict. The state is a dictionary that is passed between nodes. Each node
returns a partial dictionary of updates, which LangGraph merges into the current
state, allowing subsequent nodes to access the accumulated data.

State schema:
- query: The user's input question.
- answer: The final generated answer.
- steps: A list of strings recording the intermediate processing steps.

Nodes update state by returning a dictionary with only the keys they modify.
For example, the retrieve node adds retrieved documents to the state, while the
generate node uses the accumulated state to produce the final answer.

The workflow includes two conditional edges that allow dynamic routing based on
the query text. The first occurs after retrieval: if the query contains the word
"summary", it routes to a summarize node; otherwise it continues to the generate
node. The second occurs after generation: if the query contains the word
"complex", the answer is routed to a refine node for additional processing;
otherwise the workflow terminates. These branches demonstrate how LangGraph can
decide the next step based on state.
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


def summarize_node(state: WorkflowState) -> dict:
    """Generate a concise summary instead of a full answer."""
    return {
        "answer": f"Summary of '{state['query']}'",
        "steps": state["steps"] + ["Summarized query"],
    }


def refine_node(state: WorkflowState) -> dict:
    """Refine the answer by appending a marker and recording the step."""
    return {
        "answer": state["answer"] + " (refined)",
        "steps": state["steps"] + ["Refined answer"],
    }


def route_after_retrieve(state: WorkflowState) -> str:
    """Decide whether to summarize or generate a full answer."""
    if "summary" in state["query"].lower():
        return "summarize"
    return "generate"


def route_after_generate(state: WorkflowState) -> str:
    """Decide whether to refine the answer or end the workflow."""
    if "complex" in state["query"].lower():
        return "refine"
    return "end"


def build_graph():
    """Build and compile the LangGraph workflow."""
    graph = StateGraph(WorkflowState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("refine", refine_node)

    graph.set_entry_point("retrieve")

    # Conditional edge from retrieve to either summarize or generate.
    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {
            "summarize": "summarize",
            "generate": "generate",
        },
    )

    # Conditional edge from generate to either refine or END.
    graph.add_conditional_edges(
        "generate",
        route_after_generate,
        {
            "refine": "refine",
            "end": END,
        },
    )

    graph.add_edge("summarize", END)
    graph.add_edge("refine", END)

    return graph.compile()


def main():
    """Run the example workflow."""
    app = build_graph()

    # Simple query: retrieve -> generate -> end.
    result = app.invoke({"query": "What is LangGraph?", "steps": []})
    print(result["answer"])
    print("Steps:")
    for step in result["steps"]:
        print(f"- {step}")

    # Complex query: retrieve -> generate -> refine -> end.
    result_complex = app.invoke({"query": "Explain complex workflows", "steps": []})
    print("\nSecond run (complex query):")
    print(result_complex["answer"])
    print("Steps:")
    for step in result_complex["steps"]:
        print(f"- {step}")

    # Summary query: retrieve -> summarize -> end.
    result_summary = app.invoke({"query": "Give a summary of LangGraph", "steps": []})
    print("\nThird run (summary query):")
    print(result_summary["answer"])
    print("Steps:")
    for step in result_summary["steps"]:
        print(f"- {step}")


if __name__ == "__main__":
    main()
