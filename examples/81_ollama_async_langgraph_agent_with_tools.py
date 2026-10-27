"""Example 81: Ollama Async LangGraph Agent with Tools

+-----------------------+-----------------------------------------------------------+
| Description           | An async LangGraph agent that uses tools with Ollama.      |
| Model                 | Ollama via init_chat_model (provider-agnostic)             |
| Tools                 | Basic arithmetic tool                                      |
| Graph                 | StateGraph with tool node and routing                      |
| Async                 | Fully async node functions and invocation                 |
+-----------------------+-----------------------------------------------------------+
"""

import asyncio
import os
from operator import add
from typing import Annotated, TypedDict, Literal

from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

async def agent_node(state: AgentState) -> dict:
    """Call the Ollama model with the current message history and tools."""
    model = init_chat_model(
        model=os.getenv("OLLAMA_MODEL", "llama3.1"),
        provider="ollama",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    model_with_tools = model.bind_tools([multiply_numbers])
    response = await model_with_tools.ainvoke(state["messages"])
    return {"messages": [response]}


async def tools_node(state: AgentState) -> dict:
    """Execute any tool calls requested by the model using LangGraph's ToolNode."""
    tool_node = ToolNode([multiply_numbers])
    return await tool_node.ainvoke(state)


def router(state: AgentState) -> Literal["tools", END]:
    """Route to the tools node if the last message contains tool calls."""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Build the async LangGraph agent graph."""
    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", router)
    graph.add_edge("tools", "agent")

    return graph.compile()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

async def run_agent(question: str) -> None:
    """Run the agent in response to a user question."""
    compiled_graph = build_graph()

    messages = [
        SystemMessage(content="You are a helpful AI assistant that can use tools."),
        HumanMessage(content=question),
    ]

    result = await compiled_graph.ainvoke({"messages": messages})
    print(result["messages"][-1].content)


async def main() -> None:
    """Run a simple demonstration of the agent."""
    await run_agent("What is 7 multiplied by 8?")


if __name__ == "__main__":
    asyncio.run(main())
