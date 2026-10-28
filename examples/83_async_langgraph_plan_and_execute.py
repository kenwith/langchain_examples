"""
Async LangGraph Plan and Execute
================================

This example demonstrates an asynchronous LangGraph workflow that plans and
executes steps using a provider-agnostic chat model. The workflow first
generates a plan for a given goal, then executes each step in sequence, and
finally synthesizes the results into a single answer.

Environment Variables:
    CHAT_MODEL: The model identifier to use with init_chat_model.
    OPENAI_API_KEY: Required if using OpenAI models.
    ANTHROPIC_API_KEY: Required if using Anthropic models.
"""

import asyncio
import os
from typing import List, TypedDict

from langchain.chat_models import init_chat_model
from langgraph.graph import END, StateGraph


class PlanExecuteState(TypedDict):
    """State for the plan-and-execute workflow."""
    goal: str
    plan: List[str]
    step_index: int
    results: List[str]
    final_answer: str


def parse_plan(response: str) -> List[str]:
    """Convert a numbered list from the model into a list of step strings."""
    steps = []
    for line in response.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Remove leading numbering like "1.", "1)", "-", etc.
        if line[0].isdigit() or line[0] in "-*":
            parts = line.split(".", 1)
            if len(parts) == 2 and parts[0].strip().isdigit():
                line = parts[1].strip()
            else:
                line = line.lstrip("-* ").strip()
        if line:
            steps.append(line)
    return steps


def build_graph(model):
    """Build and return a compiled LangGraph workflow."""
    async def plan_node(state: PlanExecuteState) -> dict:
        prompt = (
            "Create a step-by-step plan to achieve the goal. "
            "Return the plan as a numbered list.\n"
            f"Goal: {state['goal']}"
        )
        response = await model.ainvoke(prompt)
        plan = parse_plan(response.content)
        return {"plan": plan, "step_index": 0, "results": []}

    async def execute_node(state: PlanExecuteState) -> dict:
        step = state["plan"][state["step_index"]]
        prompt = (
            "Execute the following step and provide the result.\n"
            f"Step: {step}"
        )
        response = await model.ainvoke(prompt)
        results = state["results"] + [response.content]
        return {"results": results, "step_index": state["step_index"] + 1}

    async def finish_node(state: PlanExecuteState) -> dict:
        combined_results = "\n".join(state["results"])
        prompt = (
            "Based on the step results, provide a final answer to the goal.\n"
            f"Goal: {state['goal']}\n"
            f"Step Results:\n{combined_results}"
        )
        response = await model.ainvoke(prompt)
        return {"final_answer": response.content}

    def should_continue(state: PlanExecuteState) -> str:
        if state["step_index"] < len(state["plan"]):
            return "execute"
        return "finish"

    workflow = StateGraph(PlanExecuteState)
    workflow.add_node("plan", plan_node)
    workflow.add_node("execute", execute_node)
    workflow.add_node("finish", finish_node)

    workflow.set_entry_point("plan")
    workflow.add_edge("plan", "execute")
    workflow.add_conditional_edges(
        "execute",
        should_continue,
        {"execute": "execute", "finish": "finish"},
    )
    workflow.add_edge("finish", END)

    return workflow.compile()


async def main() -> None:
    """Run the plan-and-execute workflow with a sample goal."""
    model_name = os.getenv("CHAT_MODEL", "openai:gpt-4o")
    model = init_chat_model(model_name, temperature=0)

    app = build_graph(model)
    goal = "Write a short story about a robot learning to paint."

    result = await app.ainvoke({"goal": goal})
    print(result["final_answer"])


if __name__ == "__main__":
    asyncio.run(main())
