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
    # Take the input_data from the state, default to empty string, and uppercase it.
    processed = state.get("input_data", "").upper()
    # Return a partial state update with the processed data.
    return {"processed_data": processed}


def validate_data(state: State) -> State:
    """Validate processed data (non-empty check)."""
    # Retrieve the processed_data from the state, default to empty string.
    processed = state.get("processed_data", "")
    # A non-empty string is considered valid.
    valid = len(processed) > 0
    # Store the boolean validation result in the state.
    return {"validation_result": valid}


def finalize(state: State) -> State:
    """Generate final output based on validation result."""
    # Check the validation_result flag; if True, build a success message.
    if state.get("validation_result"):
        final = f"Validated: {state.get('processed_data')}"
    else:
        # Otherwise, provide a failure message.
        final = "Validation failed."
    # Return the final output in the state.
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

    # --- Add nodes to the graph ---
    # Each node is a function that takes the current state and returns a partial state update.
    # The node name is used as a reference in edges and conditional edges.

    # Node: process
    # Purpose: Transform the raw input into processed data (e.g., uppercase).
    # Input state key: input_data
    # Output state key: processed_data
    workflow.add_node("process", process_data)

    # Node: validate
    # Purpose: Check the processed data and set a boolean validation result.
    # Input state key: processed_data
    # Output state key: validation_result
    workflow.add_node("validate", validate_data)

    # Node: finalize
    # Purpose: Generate the final output message based on the validation result.
    # Input state key: validation_result, processed_data
    # Output state key: final_output
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

    # Entry point: execution starts at the "process" node.
    workflow.set_entry_point("process")

    # Edge: process -> validate
    # After processing, unconditionally move to the validation step.
    workflow.add_edge("process", "validate")

    # Conditional edge: validate -> (decision)
    # The should_continue function determines which node to go to next.
    # The mapping dictionary tells the graph what each return value means.
    # In this case, "finalize" maps to the "finalize" node.
    # To support early termination, you could add:
    #   "end": END,
    # and modify should_continue to return "end" when validation fails.
    workflow.add_conditional_edges(
        "validate",
        should_continue,
        {
            "finalize": "finalize",
            # Uncomment the following line to allow early termination:
            # "end": END,
        },
    )

    # Edge: finalize -> END
    # After producing the final output, terminate the graph execution.
    workflow.add_edge("finalize", END)

    # Compile the graph into an executable app and return it.
    return workflow.compile()


def invoke_graph(initial_state: State, app: Optional[Any] = None) -> State:
    """Run the workflow using an optionally pre-built graph."""
    # If no app is provided, build one using the default build_graph function.
    if app is None:
        app = build_graph()
    # Invoke the graph with the initial state and return the final state.
    return app.invoke(initial_state)


if __name__ == "__main__":
    # Example input: a simple string to be processed by the workflow.
    initial_state: State = {"input_data": "Hello LangGraph"}

    # Run the workflow using the helper function.
    result = invoke_graph(initial_state)

    # Print the final output produced by the finalize node.
    print("Final output:", result.get("final_output"))
