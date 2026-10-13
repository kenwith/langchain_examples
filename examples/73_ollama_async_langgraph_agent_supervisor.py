"""Example 73: Async Supervisor Agent with LangGraph and Ollama

| Field    | Value                                     |
|----------|-------------------------------------------|
| Provider | Ollama                                    |
| Model    | llama3.2 (configurable via OLLAMA_MODEL)  |
| Framework| LangGraph                                 |

This example demonstrates how to build an asynchronous supervisor agent
using LangGraph and Ollama. The supervisor classifies a task and delegates
it to one of two worker nodes (summarize or translate). The model is created
with `init_chat_model` in a provider-agnostic way, using Ollama as the
backend.
"""

import asyncio
import os
from typing import Literal, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.types import Command


class AgentState(TypedDict):
    messages: list
    task: str
    next: str
    response: str


def get_llm():
    """Create a chat model using Ollama via init_chat_model."""
    model = os.getenv("OLLAMA_MODEL", "llama3.2")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    return init_chat_model(model=model, model_provider="ollama", base_url=base_url)


async def supervisor_node(
    state: AgentState,
) -> Command[Literal["summarize", "translate", "finish"]]:
    """Classify the task and route to the appropriate worker."""
    llm = get_llm()
    task = state["task"]
    prompt = (
        "You are a supervisor. Classify the following task into one of these categories:\n"
        "- summarize: if the task asks for a summary\n"
        "- translate: if the task asks for translation\n"
        "- finish: if the task is a greeting or simple request that needs no worker\n\n"
        f"Task: {task}\n\n"
        "Respond with exactly one word: summarize, translate, or finish."
    )
    messages = [SystemMessage(content=prompt), HumanMessage(content=task)]
    response = await llm.ainvoke(messages)
    decision = response.content.strip().lower()
    if decision not in ("summarize", "translate", "finish"):
        decision = "finish"
    return Command(update={"next": decision}, goto=decision)


async def summarize_node(state: AgentState) -> Command[Literal["finish"]]:
    """Summarize the input task."""
    llm = get_llm()
    task = state["task"]
    prompt = f"Summarize the following text in a few sentences:\n\n{task}"
    messages = [
        SystemMessage(content="You are a summarization assistant."),
        HumanMessage(content=prompt),
    ]
    response = await llm.ainvoke(messages)
    return Command(update={"response": response.content}, goto="finish")


async def translate_node(state: AgentState) -> Command[Literal["finish"]]:
    """Translate the input task to French."""
    llm = get_llm()
    task = state["task"]
    prompt = f"Translate the following text to French:\n\n{task}"
    messages = [
        SystemMessage(content="You are a translation assistant."),
        HumanMessage(content=prompt),
    ]
    response = await llm.ainvoke(messages)
    return Command(update={"response": response.content}, goto="finish")


async def finish_node(state: AgentState) -> dict:
    """Return the final response, or a fallback message if none was produced."""
    response = state.get("response")
    if not response:
        response = "No additional processing was performed."
    return {"response": response}


def build_graph():
    """Build and compile the LangGraph state graph."""
    graph = StateGraph(AgentState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("translate", translate_node)
    graph.add_node("finish", finish_node)
    graph.set_entry_point("supervisor")
    graph.add_edge("summarize", "finish")
    graph.add_edge("translate", "finish")
    graph.add_edge("finish", END)
    return graph.compile()


async def main():
    """Run the supervisor agent on a sample task."""
    graph = build_graph()
    sample_task = (
        "LangGraph is a library for building stateful, multi-agent applications. "
        "It provides fine-grained control over both the flow and state of your agents, "
        "making it ideal for complex, interactive systems."
    )
    result = await graph.ainvoke(
        {
            "messages": [],
            "task": sample_task,
            "next": "",
            "response": "",
        }
    )
    print(result["response"])


if __name__ == "__main__":
    asyncio.run(main())
