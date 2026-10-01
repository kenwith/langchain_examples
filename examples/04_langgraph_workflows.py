"""
Example 04: LangGraph Workflows
This example demonstrates how to build a simple stateful workflow using LangGraph.
It includes type hints for function signatures and inline comments to clarify the
state flow through the graph.
"""

from typing import Any, Optional, TypedDict
from langgraph.graph import StateGraph, END


# Define the state schema as a TypedDict
class WorkflowState(TypedDict, total=False):
    """State passed between nodes in the workflow.

    Attributes:
        input_data: Raw input string provided by the user.
        processed_data: Transformed version of input_data (e.g., uppercased).
        validation_result: Boolean indicating whether processed_data is valid.
        final_output: Final string produced after validation and finalization.
    """
    input_data: str
    processed_data: Optional[str]
    validation_result: Optional[bool]
    final_output: Optional[str]


# Type alias for brevity and readability
State = WorkflowState


# Node functions with type hints and concise docstrings
def process_data(state: State) -> State:
    """Process input data (e.g., uppercase)."""
    processed = state.get("input_data", "").upper()
    return {"processed_data": processed}


def validate_data(state: State) -> State:
    """Validate processed data (non-empty check)."""
    processed = state.get("processed_data", "")
    valid = len(processed) > 0
    return {"validation_result": valid}


def finalize(state: State) -> State:
    """Generate final output based on validation result."""
    if state.get("validation_result"):
        final = f"Validated: {state.get('processed_data')}"
    else:
        final = "Validation failed."
    return {"final_output": final}


# Conditional edge function to route based on validation
def should_continue(state: State) -> str:
    """Determine next node (always 'finalize' in this example)."""
    # For simplicity, always route to finalize. In a more complex graph,
    # this could return "finalize" or "end" based on validation_result.
    return "finalize" if state.get("validation_result") else "finalize"


# Build the graph
def build_graph() -> Any:
    """Construct and compile the LangGraph state graph."""
    workflow = StateGraph(WorkflowState)

    # Add nodes
    workflow.add_node("process", process_data)
    workflow.add_node("validate", validate_data)
    workflow.add_node("finalize", finalize)

    # --- Graph structure ---
    # The workflow follows a linear path with a conditional branch:
    #
    #   process -> validate -> (conditional) -> finalize -> END
    #
    # 1. process: transforms raw input into processed_data.
    # 2. validate: checks processed_data and sets validation_result.
    # 3. conditional edge: uses should_continue to choose the next node.
    #    In this example, it always returns "finalize", so the graph
    #    always continues to finalize. In a real application, you might
    #    return "end" to stop early or route to a different node.
    # 4. finalize: produces final_output based on validation_result.
    # 5. END: terminal node that stops execution.

    # Entry point
    workflow.set_entry_point("process")

    # process -> validate
    workflow.add_edge("process", "validate")

    # validate -> (conditional) -> finalize
    workflow.add_conditional_edges(
        "validate",
        should_continue,
        {
            "finalize": "finalize",
            # To support early termination, you could add:
            # "end": END,
            # and modify should_continue to return "end" when validation fails.
        },
    )

    # finalize -> END
    workflow.add_edge("finalize", END)

    # Compile and return
    return workflow.compile()


def invoke_graph(initial_state: State, app: Optional[Any] = None) -> State:
    """Run the workflow using an optionally pre-built graph."""
    if app is None:
        app = build_graph()
    return app.invoke(initial_state)


if __name__ == "__main__":
    # Example input
    initial_state: State = {"input_data": "Hello LangGraph"}

    # Run the workflow using the helper
    result = invoke_graph(initial_state)

    # Print the final output
    print("Final output:", result.get("final_output"))
