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
  - process -> finalize: Conditional edge based on `route_after_process`.
  - process -> END: Conditional edge when there is no processed data.
  - finalize -> END: Terminates the graph after finalization.

State flow:
- The workflow state is a TypedDict with the following keys:
  - input_data: Raw input string provided by the user.
  - processed_data: Transformed version of input_data (e.g., uppercased).
  - validation_result: Boolean indicating whether processed_data is valid.
  - final_output: Final string produced after validation and finalization.
- Each node reads only the keys it needs and returns a state update.
- Helper functions `with_processed_data` and `with_validation_result` encapsulate state updates.
- The routing helper `route_after_process` routes the graph to `finalize` or `END`
  based on whether processed data exists.

Node transition details:
- Graph execution starts at the `process` node.
- From `process`, a conditional edge invokes `route_after_process`:
  - If `processed_data` is present and non-empty, the transition goes to `finalize`.
  - Otherwise, the transition goes directly to `END`.
- From `finalize`, an unconditional edge terminates the graph at `END`.

Reusability:
- The state schema is defined as a TypedDict (`WorkflowState`), making the data contract explicit.
- The `build_graph()` function constructs and compiles the graph, allowing the workflow to be reused
  by simply calling this function whenever a new instance is needed.
- A convenience helper `create_initial_state()` is provided to initialize a state with default values.
"""

from typing import Any, Optional, TypedDict
from langgraph.graph import StateGraph, END, CompiledStateGraph


# The state schema is defined as a TypedDict. Using total=False means that not all keys
# need to be present in every state update; nodes may return only the fields they change.
# LangGraph uses this schema to validate and merge state transitions between nodes.
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


# --- Helper functions for state updates ---

def with_processed_data(state: State, processed: str) -> State:
    """Return a new state with the processed_data field set to the given value."""
    return {**state, "processed_data": processed}


def with_validation_result(state: State, valid: bool, final: str) -> State:
    """Return a new state with validation_result and final_output set."""
    return {**state, "validation_result": valid, "final_output": final}


# --- Helper functions for routing ---

def has_processed_data(state: State) -> bool:
    """Return True if processed_data is present and non-empty."""
    return bool(state.get("processed_data"))


def route_after_process(state: State) -> str:
    """Route to finalize if processed data exists, otherwise terminate."""
    return "finalize" if has_processed_data(state) else END


# Node functions with type hints and concise docstrings
def process_data(state: State) -> State:
    """Process input data (e.g., uppercase)."""
    # Read the raw input from the state; default to an empty string if absent.
    input_data = state.get("input_data", "")

    # Transform the input (e.g., uppercase) to create the processed data.
    processed = input_data.upper()

    # Write the processed data back to the state using the helper function.
    return with_processed_data(state, processed)


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

    # Write both the validation result and the final output using the helper function.
    return with_validation_result(state, valid, final)


# Build the graph
def build_graph() -> CompiledStateGraph:
    """Construct and compile the LangGraph state graph."""
    workflow = StateGraph(WorkflowState)

    # --- Add nodes to the graph ---
    # Each node is a function that takes the current state and returns a state update.
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
    # The workflow starts at the "process" node.
    #
    #   process -> (route_after_process) -> finalize -> END
    #          \-> END
    #
    # 1. process: transforms raw input into processed_data.
    # 2. route_after_process: routes to finalize if processed_data exists, otherwise END.
    # 3. finalize: checks processed_data and produces final_output.
    # 4. END: terminal node that stops execution.

    # Entry point: execution starts at the "process" node.
    # The graph always begins here, regardless of the initial state contents.
    workflow.set_entry_point("process")

    # Conditional edge: process -> finalize or END
    # The `route_after_process` function returns a key that matches one of the mapping keys.
    # LangGraph uses the returned string to select the next node from the mapping.
    # In this workflow:
    #   - "finalize" routes to the finalize node when processed_data exists.
    #   - END routes to the terminal node when processed_data is missing/empty.
    # The mapping keys must match the possible return values of the routing function.
    workflow.add_conditional_edges(
        "process",
        route_after_process,
        {
            "finalize": "finalize",
            END: END,
        },
    )

    # Edge: finalize -> END
    # After producing the final output, terminate the graph execution.
    # This is an unconditional edge because finalize always completes the workflow.
    workflow.add_edge("finalize", END)

    # Compile the graph into an executable app and return it.
    return workflow.compile()


def create_initial_state(input_data: str = "") -> WorkflowState:
    """Create a new workflow state with the given input data and default values."""
    return {
        "input_data": input_data,
        "processed_data": None,
        "validation_result": None,
        "final_output": None,
    }


def invoke_graph(initial_state: State, app: Optional[CompiledStateGraph] = None) -> State:
    """Run the workflow using an optionally pre-built graph."""
    # If no app is provided, build one using the default build_graph function.
    if app is None:
        app = build_graph()
    # Invoke the graph with the initial state and return the final state.
    return app.invoke(initial_state)


# Public API for this module
__all__ = [
    "WorkflowState",
    "State",
    "build_graph",
    "create_initial_state",
    "invoke_graph",
    "process_data",
    "finalize",
    "route_after_process",
]


if __name__ == "__main__":
    # Example input: a simple string to be processed by the workflow.
    initial_state = create_initial_state("Hello LangGraph")

    # Run the workflow using the helper function.
    result = invoke_graph(initial_state)

    # Print the final output produced by the finalize node.
    print("Final output:", result.get("final_output"))
