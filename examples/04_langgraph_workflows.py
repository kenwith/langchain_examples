"""
Example 04: LangGraph Workflows
This example demonstrates how to build a simple stateful workflow using LangGraph.
It includes type hints for function signatures and inline comments to clarify the
state flow through the graph.

Graph structure:
- Nodes:
  - process: Takes `input_data`, transforms it (uppercase), and stores it in `processed_data`.
  - finalize: Takes `processed_data`, validates it, and stores `validation_result` and `final_output`.
- Edges:
  - process -> finalize: Always after processing.
  - finalize -> END: Terminates the graph after finalization.

State flow:
- The workflow state is a TypedDict with the following keys:
  - input_data: Raw input string provided by the user.
  - processed_data: Transformed version of input_data (e.g., uppercased).
  - validation_result: Boolean indicating whether processed_data is valid.
  - final_output: Final string produced after validation and finalization.
- Each node reads only the keys it needs and returns a partial state update.
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
    # Read the raw input from the state; default to an empty string if absent.
    input_data = state.get("input_data", "")

    # Transform the input (e.g., uppercase) to create the processed data.
    processed = input_data.upper()

    # Write the processed data back to the state as a partial update.
    return {"processed_data": processed}


def finalize(state: State) -> State:
    """Generate final output based on processed data."""
    # Read the processed data from the state; default to an empty string if absent.
    processed = state.get("processed_data", "")

    # A non-empty string is considered valid.
    valid = len(processed) > 0

    # Build the final message based on validity.
    if valid:
        final = f"Validated: {processed}"
    else:
        final = "Validation failed."

    # Write both the validation result and the final output to the state.
    return {"validation_result": valid, "final_output": final}


# Build the graph
def build_graph() -> Any:
    """Construct and compile the LangGraph state graph."""
    workflow = StateGraph(WorkflowState)

    # --- Add nodes to the graph ---
    # Each node is a function that takes the current state and returns a partial state update.
    # The node name is used as a reference in edges.

    # Node: process
    # Purpose: Transform the raw input into processed data (e.g., uppercase).
    # Input state key: input_data
    # Output state key: processed_data
    workflow.add_node("process", process_data)

    # Node: finalize
    # Purpose: Validate the processed data and generate the final output.
    # Input state key: processed_data
    # Output state keys: validation_result, final_output
    workflow.add_node("finalize", finalize)

    # --- Graph structure ---
    # The workflow is a simple linear path with two nodes:
    #
    #   process -> finalize -> END
    #
    # 1. process: transforms raw input into processed_data.
    # 2. finalize: checks processed_data and produces final_output.
    # 3. END: terminal node that stops execution.

    # Entry point: execution starts at the "process" node.
    workflow.set_entry_point("process")

    # Edge: process -> finalize
    # After processing, unconditionally move to the finalization step.
    workflow.add_edge("process", "finalize")

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
