"""
Async LangGraph Agent with Ollama
=================================

| Example | Description | Dependencies |
|---------|-------------|--------------|
| 75 | Async LangGraph agent using Ollama | langchain, langgraph, ollama |
"""

import asyncio
import os
from typing import Annotated, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


@tool
def add_numbers(a: int, b: int) -> int:
    """Add two integers and return the sum."""
    return a + b


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


async def agent_node(state: AgentState) -> dict:
    """Invoke the model with the current message history."""
    response = await model_with_tools.ainvoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """Determine whether to call tools or finish."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "end"


tools = [add_numbers]
model = init_chat_model(
    model=os.getenv("OLLAMA_MODEL", "llama3.1"),
    model_provider="ollama",
    temperature=0,
)
model_with_tools = model.bind_tools(tools)

tool_node = ToolNode(tools)

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", "end": END},
)
workflow.add_edge("tools", "agent")

app = workflow.compile()


async def main() -> None:
    """Run a simple query against the async LangGraph agent."""
    result = await app.ainvoke(
        {
            "messages": [
                SystemMessage(
                    content="You are a helpful assistant. Use the provided tools to answer questions."
                ),
                HumanMessage(content="What is 2 + 3?"),
            ]
        }
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
