"""LangGraph Agent Supervisor.

+---------------------+--------------------------------------------------+
| Field               | Value                                            |
+---------------------+--------------------------------------------------+
| Example             | 33                                               |
| Title               | LangGraph Agent Supervisor                       |
| Description         | A supervisor delegates to research and writer    |
|                     | agents using LangGraph and init_chat_model.      |
| Model               | Provider-agnostic (OpenAI, Anthropic, etc.)      |
| Environment         | CHAT_MODEL, CHAT_MODEL_PROVIDER, and the API key |
|                     | for the selected provider.                       |
+---------------------+--------------------------------------------------+

This example builds a small LangGraph graph with three nodes:

- supervisor: decides whether to run research, writer, or finish.
- research: performs research on the task.
- writer: writes the final article using the research.

The supervisor uses a simple state-based rule: if no research has
been produced, route to research; if research exists but no article,
route to writer; otherwise finish.

Run:

    python examples/33_langgraph_agent_supervisor.py
"""

from __future__ import annotations

import operator
import os
from typing import Annotated, Literal, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph


class AgentState(TypedDict):
    """State of the agent supervisor graph."""

    messages: Annotated[list, operator.add]
    task: str
    next: Literal["research", "writer", "finish"]


def get_model():
    """Initialize a chat model from environment variables."""
    return init_chat_model(
        model=os.getenv("CHAT_MODEL", "gpt-4o"),
        model_provider=os.getenv("CHAT_MODEL_PROVIDER"),
        temperature=0,
    )


def supervisor_node(state: AgentState) -> dict:
    """Decide which agent should act next based on the current state."""
    has_research = any(
        isinstance(m, AIMessage) and m.content.startswith("RESEARCH:")
        for m in state["messages"]
    )
    has_article = any(
        isinstance(m, AIMessage) and m.content.startswith("ARTICLE:")
        for m in state["messages"]
    )

    if not has_research:
        return {"next": "research"}
    elif not has_article:
        return {"next": "writer"}
    else:
        return {"next": "finish"}


def research_node(state: AgentState) -> dict:
    """Perform research on the task and append a research message."""
    model = get_model()
    system = SystemMessage(
        content="You are a research agent. Provide a concise research summary."
    )
    user = HumanMessage(content=state["task"])
    response = model.invoke([system, user])
    return {"messages": [AIMessage(content=f"RESEARCH:\n{response.content}")]}


def writer_node(state: AgentState) -> dict:
    """Write a final article using the research and append an article message."""
    model = get_model()
    research = next(
        (
            m.content
            for m in reversed(state["messages"])
            if m.content.startswith("RESEARCH:")
        ),
        "No research available.",
    )
    system = SystemMessage(
        content="You are a writer agent. Write a well-structured article based on the research."
    )
    user = HumanMessage(content=f"Task: {state['task']}\n\nResearch:\n{research}")
    response = model.invoke([system, user])
    return {"messages": [AIMessage(content=f"ARTICLE:\n{response.content}")]}


def build_graph() -> CompiledStateGraph:
    """Build and compile the supervisor graph."""
    graph = StateGraph(AgentState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("research", research_node)
    graph.add_node("writer", writer_node)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
        {"research": "research", "writer": "writer", "finish": END},
    )
    graph.add_edge("research", "supervisor")
    graph.add_edge("writer", "supervisor")

    return graph.compile()


if __name__ == "__main__":
    task = "Write a short article about LangGraph."
    graph = build_graph()
    result = graph.invoke({"task": task, "messages": [], "next": "research"})

    print("Final messages:")
    for msg in result["messages"]:
        print(f"{msg.type}: {msg.content}")
