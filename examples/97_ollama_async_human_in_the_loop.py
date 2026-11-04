"""Ollama async human-in-the-loop example using LangGraph.

This example demonstrates an interrupt-based approval step in an async LangGraph
workflow. The graph generates a draft response with a local Ollama model, pauses
for human approval, and only finalizes the response after approval.
"""

# ---------------------------------------------------------------------------
# Example 97: Ollama Async Human-in-the-Loop
# ---------------------------------------------------------------------------
# | Field        | Value                                                    |
# |--------------|----------------------------------------------------------|
# | Description  | Async LangGraph workflow with Ollama and approval step    |
# | Dependencies | langchain, langgraph, langchain-ollama, asyncio           |
# | Model        | OLLAMA_MODEL (default: llama3.2)                          |
# | Base URL     | OLLAMA_BASE_URL (default: http://localhost:11434)         |
# ---------------------------------------------------------------------------

import asyncio
import os
from typing import TypedDict

from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph


class AgentState(TypedDict):
    draft: str
    approved: bool
    final: str


async def generate_draft(state: AgentState) -> dict:
    """Generate a draft response using a local Ollama model."""
    model = init_chat_model(
        os.getenv("OLLAMA_MODEL", "llama3.2"),
        model_provider="ollama",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    response = await model.ainvoke("Write a short greeting.")
    return {"draft": response.content}


async def approval_node(state: AgentState) -> dict:
    """Pause for human approval before finalizing."""
    print(f"\nDraft response:\n{state['draft']}\n")
    answer = await asyncio.to_thread(input, "Approve this response? (y/n): ")
    return {"approved": answer.strip().lower() == "y"}


async def finalize_response(state: AgentState) -> dict:
    """Finalize the approved response."""
    return {"final": f"Finalized: {state['draft']}"}


def build_graph():
    """Build and compile the LangGraph workflow."""
    builder = StateGraph(AgentState)
    builder.add_node("generate_draft", generate_draft)
    builder.add_node("approval", approval_node)
    builder.add_node("finalize", finalize_response)

    builder.add_edge(START, "generate_draft")
    builder.add_edge("generate_draft", "approval")
    builder.add_conditional_edge("approval", "finalize", lambda state: state["approved"])
    builder.add_conditional_edge("approval", END, lambda state: not state["approved"])

    return builder.compile()


async def main() -> None:
    """Run the async human-in-the-loop workflow."""
    graph = build_graph()
    result = await graph.ainvoke(
        {"draft": "", "approved": False, "final": ""}
    )
    if result.get("final"):
        print(result["final"])
    else:
        print("Response not approved.")


if __name__ == "__main__":
    asyncio.run(main())
