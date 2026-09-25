"""
Example 04: LangGraph Workflows
This example demonstrates how to build a simple stateful workflow using LangGraph.
It includes type hints for function signatures and inline comments to clarify the
state flow through the graph.
"""

from typing import Dict, List, Any, TypedDict, Optional, Callable
from langgraph.graph import StateGraph, END
from langgraph.graph.state import StateGraph as SG


# Define the state schema as a TypedDict
class WorkflowState(TypedDict, total=False):
    """State passed between nodes in the workflow."""
    input_data: str
    processed_data: Optional[str]
    validation_result: Optional[bool]
    final_output: Optional[str]


# Node functions with type hints and comments
def process_data(state: WorkflowState) -> WorkflowState:
    """Process the input data (e.g., normalize, transform).

    Args:
        state: Current workflow state.

    Returns:
        Updated state with processed_data field set.
    """
    # Simulate processing by uppercasing input
    processed = state.get("input_data", "").upper()
    return {"processed_data": processed}


def validate_data(state: WorkflowState) -> WorkflowState:
    """Validate the processed data.

    Args:
        state: Current workflow state.

    Returns:
        Updated state with validation_result.
    """
    # Simple validation: check if processed data is not empty
    processed = state.get("processed_data", "")
    valid = len(processed) > 0
    return {"validation_result": valid}


def finalize(state: WorkflowState) -> WorkflowState:
    """Generate final output based on validation.

    Args:
        state: Current workflow state.

    Returns:
        Updated state with final_output.
    """
    if state.get("validation_result"):
        final = f"Validated: {state.get('processed_data')}"
    else:
        final = "Validation failed."
    return {"final_output": final}


# Conditional edge function to route based on validation
def should_continue(state: WorkflowState) -> str:
    """Determine next node based on validation result.

    Args:
        state: Current workflow state.

    Returns:
        String indicating next node name.
    """
    return "finalize" if state.get("validation_result") else "finalize"  # For simplicity, always finalize


# Build the graph
def build_workflow() -> StateGraph:
    """Construct the LangGraph state graph.

    Returns:
        Compiled graph ready for execution.
    """
    # Initialize graph with state schema
    workflow = StateGraph(WorkflowState)

    # Add nodes
    workflow.add_node("process", process_data)
    workflow.add_node("validate", validate_data)
    workflow.add_node("finalize", finalize)

    # Define edges: start -> process -> validate -> conditional -> finalize -> end
    workflow.set_entry_point("process")
    workflow.add_edge("process", "validate")
    # Use conditional edge from validate to finalize (or end, but we always go to finalize)
    workflow.add_conditional_edges(
        "validate",
        should_continue,
        {
            "finalize": "finalize",
            # If we had an end node, we could route there; here we always finalize
        },
    )
    workflow.add_edge("finalize", END)

    # Compile and return
    return workflow.compile()


if __name__ == "__main__":
    # Build the graph
    app = build_workflow()

    # Example input
    initial_state: WorkflowState = {"input_data": "Hello LangGraph"}

    # Run the workflow
    result = app.invoke(initial_state)

    # Print the final output
    print("Final output:", result.get("final_output"))
