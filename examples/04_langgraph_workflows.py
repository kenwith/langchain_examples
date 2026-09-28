"""
Example 04: LangGraph Workflows
This example demonstrates how to build a simple stateful workflow using LangGraph.
It includes type hints for function signatures and inline comments to clarify the
state flow through the graph.
"""

from typing import Dict, List, Any, TypedDict, Optional, Callable
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


# Node functions with type hints and comments
def process_data(state: WorkflowState) -> WorkflowState:
    """Process the input data (e.g., normalize, transform).

    Args:
        state: Current workflow state.

    Returns:
        Updated state with processed_data field set.
    """
    # Simulate processing by uppercasing input.
    # The returned dict is a partial state update; LangGraph merges it
    # into the current state, so other fields remain unchanged.
    processed = state.get("input_data", "").upper()
    return {"processed_data": processed}


def validate_data(state: WorkflowState) -> WorkflowState:
    """Validate the processed data.

    Args:
        state: Current workflow state.

    Returns:
        Updated state with validation_result.
    """
    # Simple validation: check if processed data is not empty.
    # This node receives the state after process_data, so processed_data
    # is expected to be set by the time this node runs.
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
    # This node runs after the conditional edge, so validation_result
    # should be available from the validate_data node.
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
    # For simplicity, always route to finalize. In a more complex graph,
    # this could return "finalize" or "end" based on validation_result.
    return "finalize" if state.get("validation_result") else "finalize"


# Build the graph
def build_workflow() -> StateGraph:
    """Construct the LangGraph state graph.

    Returns:
        Compiled graph ready for execution.
    """
    # Initialize graph with state schema
    workflow = StateGraph(WorkflowState)

    # Add nodes. Each node is a function that takes the current state
    # and returns a partial state update to merge into the graph state.
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

    # Entry point: process is the first node to execute.
    workflow.set_entry_point("process")

    # process -> validate: after processing, always validate.
    workflow.add_edge("process", "validate")

    # validate -> (conditional) -> finalize
    # The conditional edge uses should_continue to determine the next node.
    # The mapping keys are the possible return values from should_continue;
    # the values are the node names to route to.
    # Here, should_continue always returns "finalize", so the graph always
    # proceeds to finalize after validation.
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

    # finalize -> END: after final output is generated, stop.
    workflow.add_edge("finalize", END)

    # Compile and return
    return workflow.compile()


def run_workflow(initial_state: WorkflowState) -> WorkflowState:
    """Build and run the workflow, returning the final state.

    Args:
        initial_state: Starting state for the workflow.

    Returns:
        Final state after graph execution.
    """
    app = build_workflow()
    return app.invoke(initial_state)


if __name__ == "__main__":
    # Example input
    initial_state: WorkflowState = {"input_data": "Hello LangGraph"}

    # Run the workflow using the helper
    result = run_workflow(initial_state)

    # Print the final output
    print("Final output:", result.get("final_output"))
