"""
Example 04: LangGraph Workflows with Conditional Edges

This example demonstrates how to use conditional edges in a LangGraph graph.
The graph represents a simple agent that decides whether to perform additional
research or answer the query directly based on the number of steps already taken.

The workflow is structured as a cycle:
- The router node evaluates the current state and conditionally routes to either
  the research node, the answer node, or a failure node.
- The research node increments the step counter and appends a research note to
  the answer, then returns to the router for another decision.
- The answer node appends a final answer and terminates the graph.
- The failure node produces a clear message when the step limit is exceeded.

The graph includes a step-limit guard: if the number of steps reaches a hard
maximum (``max_total_steps``), the workflow routes to the failure node instead
of continuing. This prevents infinite loops and provides a clear error message.

This file also provides a reusable ``run_workflow()`` helper that builds the
graph, runs it with a given initial state, and returns the final state. This
makes it easy to experiment with different queries and step limits.
"""

from typing import TypedDict, Literal, Optional
from langgraph.graph import StateGraph, END


class AgentState(TypedDict):
    """State dictionary for the agent.

    Attributes:
        query: The user's original question.
        steps: The number of research steps already taken.
        max_steps: The maximum number of research steps allowed before answering.
        max_total_steps: The absolute maximum number of steps allowed before failing.
        answer: The accumulated answer text built by the nodes.
        status: The final status of the workflow: "success" or "failed".
    """
    query: str
    steps: int
    max_steps: int
    max_total_steps: int
    answer: str
    status: str


def router_node(state: AgentState) -> AgentState:
    """Route the workflow to the next appropriate node.

    In a real application, this node could use an LLM to decide the next
    action. For this example, it simply passes the state through unchanged;
    the actual routing is performed by the conditional edge function
    :func:`should_continue`.

    Args:
        state: The current agent state.

    Returns:
        The same state, unmodified.
    """
    return state


def research_node(state: AgentState) -> AgentState:
    """Perform one research step and update the agent state.

    This node increments the ``steps`` counter and appends a short research
    note to the ``answer`` field. The updated state is returned so that the
    router can decide whether more research is needed.

    Args:
        state: The current agent state.

    Returns:
        A new state with ``steps`` incremented and ``answer`` extended.
    """
    return {
        "query": state["query"],
        "steps": state["steps"] + 1,
        "max_steps": state["max_steps"],
        "max_total_steps": state["max_total_steps"],
        "answer": state["answer"] + f" Research step {state['steps'] + 1};",
        "status": state["status"],
    }


def answer_node(state: AgentState) -> AgentState:
    """Produce the final answer and terminate the workflow.

    This node appends a final answer marker to the ``answer`` field and sets
    the status to "success". It does not increment the step counter. The graph
    ends after this node runs.

    Args:
        state: The current agent state.

    Returns:
        A new state with the final answer appended and status set to "success".
    """
    return {
        "query": state["query"],
        "steps": state["steps"],
        "max_steps": state["max_steps"],
        "max_total_steps": state["max_total_steps"],
        "answer": state["answer"] + " Final answer.",
        "status": "success",
    }


def failed_node(state: AgentState) -> AgentState:
    """Handle the failure case when the step limit is exceeded.

    This node appends a clear failure message to the ``answer`` field and sets
    the status to "failed". The graph ends after this node runs.

    Args:
        state: The current agent state.

    Returns:
        A new state with a failure message and status set to "failed".
    """
    return {
        "query": state["query"],
        "steps": state["steps"],
        "max_steps": state["max_steps"],
        "max_total_steps": state["max_total_steps"],
        "answer": state["answer"] + " FAILED: Step limit exceeded.",
        "status": "failed",
    }


def should_continue(state: AgentState) -> Literal["research", "answer", "failed"]:
    """Decide whether to continue researching, answer, or fail.

    This conditional edge function is attached to the router node. It checks
    the current number of steps against two limits:
    - ``max_total_steps``: If the step count reaches this hard limit, the
      workflow routes to the ``"failed"`` node to prevent infinite loops.
    - ``max_steps``: If the step count reaches this soft limit, the workflow
      routes to the ``"answer"`` node to produce a final answer.
    Otherwise, it routes to the ``"research"`` node for another step.

    Args:
        state: The current agent state.

    Returns:
        The name of the next node to execute: ``"research"``, ``"answer"``,
        or ``"failed"``.
    """
    if state["steps"] >= state["max_total_steps"]:
        return "failed"
    elif state["steps"] >= state["max_steps"]:
        return "answer"
    else:
        return "research"


def run_workflow(initial_state: Optional[AgentState] = None) -> AgentState:
    """Build and run the LangGraph workflow.

    This helper constructs the state graph, wires up the nodes and edges, and
    invokes the graph with the provided initial state. If no initial state is
    given, a default sample state is used.

    Args:
        initial_state: Optional dictionary containing the initial values for
            ``query``, ``steps``, ``max_steps``, ``max_total_steps``, ``answer``,
            and ``status``. If ``None``, a default query and step limits are used.

    Returns:
        The final state dictionary after the graph has completed, including
        the accumulated answer, the total number of steps taken, and the status.
    """
    if initial_state is None:
        initial_state = {
            "query": "What is LangGraph?",
            "steps": 0,
            "max_steps": 3,
            "max_total_steps": 5,  # Hard limit to prevent infinite loops
            "answer": "",
            "status": "in_progress",
        }

    # Build the state graph
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("router", router_node)
    graph.add_node("research", research_node)
    graph.add_node("answer", answer_node)
    graph.add_node("failed", failed_node)

    # Set entry point
    graph.set_entry_point("router")

    # Add conditional edges
    graph.add_conditional_edges(
        "router",
        should_continue,
        {
            "research": "research",
            "answer": "answer",
            "failed": "failed",
        }
    )

    # Add normal edges
    graph.add_edge("research", "router")
    graph.add_edge("answer", END)
    graph.add_edge("failed", END)

    # Compile the graph
    app = graph.compile()

    # Run the graph
    result = app.invoke(initial_state)

    return result


def main():
    """Run the example workflow and print the results.

    This function calls :func:`run_workflow` with the default sample state and
    prints the query, the number of steps taken, the status, and the final answer.
    """
    result = run_workflow()

    print(f"Query: {result['query']}")
    print(f"Steps: {result['steps']}")
    print(f"Status: {result['status']}")
    print(f"Answer: {result['answer']}")


if __name__ == "__main__":
    main()
