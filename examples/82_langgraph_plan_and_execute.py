"""
Example 82: LangGraph Plan and Execute

| Parameter | Value |
|-----------|-------|
| Model | Any chat model supported by init_chat_model |
| Temperature | 0.0 |
| Model env var | CHAT_MODEL |

This example defines a small plan-and-execute workflow with LangGraph.
The `build_graph` function exposes the compiled graph, and the planner
and executor nodes are standalone functions that can be tested in isolation.
"""

import logging
import os
import time
from typing import TypedDict, List

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END

logger = logging.getLogger(__name__)

__all__ = [
    "PlanExecuteState",
    "get_chat_model",
    "format_plan",
    "planner_node",
    "executor_node",
    "build_graph",
    "run_plan_and_execute",
]


class PlanExecuteState(TypedDict):
    request: str
    plan: List[str]
    results: List[str]


def get_chat_model():
    """Initialize a chat model using the CHAT_MODEL environment variable."""
    return init_chat_model(
        model=os.getenv("CHAT_MODEL", "gpt-4o"),
        temperature=0.0,
    )


def format_plan(plan: List[str]) -> str:
    """Format a plan into a numbered, human-readable string."""
    return "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))


def planner_node(state: PlanExecuteState) -> dict:
    """
    Generate a step-by-step plan for the request in the state.

    This node is designed to be invoked directly with a `PlanExecuteState`
    mapping for isolated testing, or wired into a LangGraph workflow via
    `build_graph`.

    Args:
        state: A mapping that must contain a "request" key with the user request.

    Returns:
        A dictionary with a single "plan" key containing a list of plan steps.

    Testing:
        The chat model can be configured by setting the CHAT_MODEL environment
        variable before calling this node. For deterministic tests, monkeypatch
        `get_chat_model` to return a fake model that returns a fixed plan.
    """
    model = get_chat_model()
    prompt = (
        f"Create a step-by-step plan to address the request: {state['request']}\n"
        "Return each step on a new line."
    )
    response = model.invoke(prompt)
    plan = [line.strip() for line in response.splitlines() if line.strip()]
    return {"plan": plan}


def executor_node(state: PlanExecuteState) -> dict:
    """
    Execute each step in the state's plan and collect the results.

    This node is designed to be invoked directly with a `PlanExecuteState`
    mapping for isolated testing, or wired into a LangGraph workflow via
    `build_graph`.

    Args:
        state: A mapping containing "request", "plan", and optionally "results".

    Returns:
        A dictionary with a single "results" key containing the output of each
        executed step.

    Testing:
        The chat model can be configured by setting the CHAT_MODEL environment
        variable before calling this node. For deterministic tests, monkeypatch
        `get_chat_model` to return a fake model that returns predictable text.
    """
    model = get_chat_model()
    results = []
    plan_text = format_plan(state["plan"])
    for step in state["plan"]:
        prompt = (
            f"Plan:\n{plan_text}\n\n"
            f"Execute the following step for the request '{state['request']}':\n"
            f"{step}"
        )
        start_time = time.perf_counter()
        try:
            result = model.invoke(prompt)
        finally:
            elapsed = time.perf_counter() - start_time
            logger.info("Executed step: %s (%.2f seconds)", step, elapsed)
        results.append(result)
    return {"results": results}


def build_graph():
    """
    Build and compile the plan-and-execute LangGraph workflow.

    The graph has two nodes:
        - planner: generates a plan from the request.
        - executor: executes each plan step.

    Returns:
        A compiled `CompiledStateGraph` that accepts an initial state dictionary
        with a "request" key and returns the completed state with "plan" and
        "results".

    Testing:
        The returned graph can be exercised directly with `graph.invoke(...)`.
        Alternatively, each node can be unit-tested independently by calling it
        with a state dictionary.
    """
    graph = StateGraph(PlanExecuteState)
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", END)
    return graph.compile()


def run_plan_and_execute(request: str) -> dict:
    """Run the plan-and-execute workflow for a given request."""
    app = build_graph()
    return app.invoke({"request": request})


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    request = "Plan a birthday party for a 10-year-old."
    output = run_plan_and_execute(request)
    print("Request:", request)
    print("Plan:")
    print(format_plan(output["plan"]))
    print("Results:")
    for result in output["results"]:
        print(f"- {result}")
