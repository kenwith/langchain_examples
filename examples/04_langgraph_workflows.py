"""
Example 04: LangGraph Workflows with Conditional Edges

This example demonstrates how to use conditional edges in a LangGraph graph.
The graph represents a simple agent that decides whether to perform additional
research or answer the query directly based on the number of steps already taken.

The workflow is structured as a cycle:
- The router node evaluates the current state and conditionally routes to either
  the research node or the answer node.
- The research node increments the step counter and appends a research note to
  the answer, then returns to the router for another decision.
- The answer node appends a final answer and terminates the graph.

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
        max_steps: The maximum number of research steps allowed.
        answer: The accumulated answer text built by the nodes.
    """
    query: str
    steps: int
    max_steps: int
    answer: str


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
        "answer": state["answer"] + f" Research step {state['steps'] + 1};",
    }


def answer_node(state: AgentState) -> AgentState:
    """Produce the final answer and terminate the workflow.

    This node appends a final answer marker to the ``answer`` field. It does
    not increment the step counter. The graph ends after this node runs.

    Args:
        state: The current agent state.

    Returns:
        A new state with the final answer appended.
    """
    return {
        "query": state["query"],
        "steps": state["steps"],
        "max_steps": state["max_steps"],
        "answer": state["answer"] + " Final answer.",
    }


def should_continue(state: AgentState) -> Literal["research", "answer"]:
    """Decide whether to continue researching or answer directly.

    This conditional edge function is attached to the router node. It checks
    the current number of steps against ``max_steps``. If the step limit has
    not been reached, the workflow routes to the ``"research"`` node;
    otherwise it routes to the ``"answer"`` node.

    Args:
        state: The current agent state.

    Returns:
        The name of the next node to execute: either ``"research"`` or
        ``"answer"``.
    """
    if state["steps"] < state["max_steps"]:
        return "research"
    else:
        return "answer"


def run_workflow(initial_state: Optional[AgentState] = None) -> AgentState:
    """Build and run the LangGraph workflow.

    This helper constructs the state graph, wires up the nodes and edges, and
    invokes the graph with the provided initial state. If no initial state is
    given, a default sample state is used.

    Args:
        initial_state: Optional dictionary containing the initial values for
            ``query``, ``steps``, ``max_steps``, and ``answer``. If ``None``,
            a default query and step limit are used.

    Returns:
        The final state dictionary after the graph has completed, including
        the accumulated answer and the total number of steps taken.
    """
    if initial_state is None:
        initial_state = {
            "query": "What is LangGraph?",
            "steps": 0,
            "max_steps": 3,
            "answer": "",
        }

    # Build the state graph
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("router", router_node)
    graph.add_node("research", research_node)
    graph.add_node("answer", answer_node)

    # Set entry point
    graph.set_entry_point("router")

    # Add conditional edges
    graph.add_conditional_edges(
        "router",
        should_continue,
        {
            "research": "research",
            "answer": "answer",
        }
    )

    # Add normal edges
    graph.add_edge("research", "router")
    graph.add_edge("answer", END)

    # Compile the graph
    app = graph.compile()

    # Run the graph
    result = app.invoke(initial_state)

    return result


def main():
    """Run the example workflow and print the results.

    This function calls :func:`run_workflow` with the default sample state and
    prints the query, the number of steps taken, and the final answer.
    """
    result = run_workflow()

    print(f"Query: {result['query']}")
    print(f"Steps: {result['steps']}")
    print(f"Answer: {result['answer']}")


if __name__ == "__main__":
    main()
