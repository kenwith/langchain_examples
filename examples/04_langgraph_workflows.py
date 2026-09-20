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
graph, compiles it, runs it with a given initial state, and returns the final
state. This makes it easy to experiment with different queries and step limits.

State Flow
----------
The agent state is a TypedDict that is passed from node to node. Each node
returns a dictionary (typically only the fields it modifies) that gets merged
into the current state by LangGraph. This makes it easy to add new fields or
modify existing ones without affecting unrelated parts of the state.

The data flow is best understood by following the ``steps`` counter:

1. The router node reads the current ``steps`` value and the step limits.
2. If ``steps`` is below the soft limit ``max_steps``, the graph goes to the
   ``research`` node, which increments ``steps`` and appends to the answer.
3. If ``steps`` reaches ``max_steps`` but not ``max_total_steps``, the graph
   goes to the ``answer`` node, which appends a final answer and sets status.
4. If ``steps`` reaches ``max_total_steps``, the graph goes to the ``failed``
   node, which appends an error message and sets status to "failed".

This modular structure makes it straightforward to add new node types (e.g.,
a "search" node) by simply adding a new function, adding it to the graph, and
adjusting the routing logic in ``should_continue``.
"""

from typing import Literal, Optional, TypedDict

from langgraph.graph import StateGraph, END


class AgentState(TypedDict):
    """Typed state dictionary for the agent.

    Attributes:
        query: The user's original question.
        steps: The number of research steps already taken.
        max_steps: The maximum number of research steps allowed before answering.
        max_total_steps: The absolute maximum number of steps allowed before failing.
        answer: The accumulated answer text built by the nodes.
        status: The final status of the workflow: "success" or "failed".
    """
    # Data that is read by the router and used for control flow.
    query: str
    steps: int
    max_steps: int
    max_total_steps: int

    # Data that is modified over the course of the workflow.
    answer: str
    status: str


def router_node(state: AgentState) -> AgentState:
    """Route the workflow to the next appropriate node.

    In a real application, this node could use an LLM to decide the next
    action. For this example, it simply passes the state through unchanged;
    the actual routing is performed by the conditional edge function
    :func:`should_continue`.

    Reads: ``steps``, ``max_steps``, ``max_total_steps`` (indirectly via the
           conditional edge function).
    Updates: None (returns the same state).

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

    Reads: ``query``, ``steps``, ``max_steps``, ``max_total_steps``, ``answer``,
           ``status``.
    Updates: ``steps`` (incremented by 1), ``answer`` (extended with a note).

    Args:
        state: The current agent state.

    Returns:
        A partial state update with ``steps`` incremented and ``answer`` extended.
    """
    return {
        "steps": state["steps"] + 1,
        "answer": state["answer"] + f" Research step {state['steps'] + 1};",
    }


def answer_node(state: AgentState) -> AgentState:
    """Produce the final answer and terminate the workflow.

    This node appends a final answer marker to the ``answer`` field and sets
    the status to "success". It does not increment the step counter. The graph
    ends after this node runs.

    Reads: ``answer`` (current accumulated text).
    Updates: ``answer`` (appends final answer), ``status`` (sets to "success").

    Args:
        state: The current agent state.

    Returns:
        A partial state update with the final answer and status "success".
    """
    return {
        "answer": state["answer"] + " Final answer.",
        "status": "success",
    }


def failed_node(state: AgentState) -> AgentState:
    """Handle the failure case when the step limit is exceeded.

    This node appends a clear failure message to the ``answer`` field and sets
    the status to "failed". The graph ends after this node runs.

    Reads: ``answer`` (current accumulated text).
    Updates: ``answer`` (appends failure message), ``status`` (sets to "failed").

    Args:
        state: The current agent state.

    Returns:
        A partial state update with a failure message and status "failed".
    """
    return {
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

    Reads: ``steps``, ``max_total_steps``, ``max_steps``.
    Updates: None (returns a routing decision).

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


def build_agent_graph() -> StateGraph:
    """Build the LangGraph state graph for the agent workflow.

    The graph contains a router node with conditional edges to research,
    answer, and failed nodes. The research node loops back to the router,
    while the answer and failed nodes terminate at the END node.

    Graph layout::

        +----------+
        |  router  |
        +----------+
            |
            | conditional (should_continue)
            |                  |
      +------+------+----------+
      |             |          |
      v             v          v
  research      answer      failed
      |             |          |
      +-------------+          |
      | (loop back)            |
      +------------------------+ --> END

    When compiled, this graph will execute the router, then conditionally
    send the state to one of the three terminal/loop nodes. If research runs,
    it returns to the router for another cycle.

    Returns:
        A fully wired :class:`StateGraph` ready to be compiled.
    """
    graph = StateGraph(AgentState)

    # Add nodes. The node functions receive the full state and return a
    # partial update that is merged back into the state.
    graph.add_node("router", router_node)     # Routes based on step count
    graph.add_node("research", research_node) # Increments steps, adds research note
    graph.add_node("answer", answer_node)     # Appends final answer, status = success
    graph.add_node("failed", failed_node)     # Appends error, status = failed

    # Set the entry point: the graph always starts at the router.
    graph.set_entry_point("router")

    # Conditional edges from the router.
    # The should_continue function returns the name of the next node,
    # and the mapping tells LangGraph how to interpret those names.
    graph.add_conditional_edges(
        "router",
        should_continue,
        {
            "research": "research",
            "answer": "answer",
            "failed": "failed",
        }
    )

    # Normal (unconditional) edges:
    # - research node returns to the router for another decision cycle.
    # - answer and failed nodes terminate the graph.
    graph.add_edge("research", "router")
    graph.add_edge("answer", END)
    graph.add_edge("failed", END)

    return graph


def run_workflow(initial_state: Optional[AgentState] = None) -> AgentState:
    """Build, compile, and run the LangGraph workflow.

    This helper constructs the state graph via :func:`build_agent_graph`,
    compiles it into an executable application, and invokes it with the
    provided initial state. If no initial state is given, a default sample
    state is used.

    Args:
        initial_state: Optional :class:`AgentState` containing the initial values
            for the workflow. If ``None``, a default query and step limits are used.

    Returns:
        The final :class:`AgentState` after the graph has completed, including
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
    graph = build_agent_graph()

    # Compile the graph into an executable application
    app = graph.compile()

    # Run the graph with the initial state
    final_state = app.invoke(initial_state)

    return final_state


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
