"""
Example 04: LangGraph Workflows with Conditional Edges

This example demonstrates how to use conditional edges in a LangGraph graph.
The graph represents a simple agent that decides whether to perform additional
research or answer the query directly based on the number of steps already taken.
"""

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END


class AgentState(TypedDict):
    """The state of the agent."""
    query: str
    steps: int
    max_steps: int
    answer: str


def router_node(state: AgentState) -> AgentState:
    """Node that decides the next step."""
    # In a real application, this could be an LLM call.
    # For this example, we just pass the state through.
    return state


def research_node(state: AgentState) -> AgentState:
    """Node that performs research."""
    return {
        "query": state["query"],
        "steps": state["steps"] + 1,
        "max_steps": state["max_steps"],
        "answer": state["answer"] + f" Research step {state['steps'] + 1};",
    }


def answer_node(state: AgentState) -> AgentState:
    """Node that provides a final answer."""
    return {
        "query": state["query"],
        "steps": state["steps"],
        "max_steps": state["max_steps"],
        "answer": state["answer"] + " Final answer.",
    }


def should_continue(state: AgentState) -> Literal["research", "answer"]:
    """Conditional edge function that routes to research or answer."""
    if state["steps"] < state["max_steps"]:
        return "research"
    else:
        return "answer"


def main():
    """Build and run the graph."""
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
    result = app.invoke(
        {
            "query": "What is LangGraph?",
            "steps": 0,
            "max_steps": 3,
            "answer": "",
        }
    )

    print(f"Query: {result['query']}")
    print(f"Steps: {result['steps']}")
    print(f"Answer: {result['answer']}")


if __name__ == "__main__":
    main()
