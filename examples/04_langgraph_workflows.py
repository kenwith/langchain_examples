import os
import sqlite3
from typing import Annotated, List, Literal, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

DB_PATH = "checkpoints.sqlite"

# The shared state passed between nodes.
# `messages` accumulates over time via the reducer (append), so each step sees the full conversation.
# `user_id` and `thread_id` are metadata used for checkpointing and resuming sessions.
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    user_id: str
    thread_id: str

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    weather_data = {
        "new york": "Sunny, 72°F",
        "london": "Cloudy, 58°F",
        "tokyo": "Rainy, 65°F",
        "paris": "Partly cloudy, 68°F",
        "sydney": "Clear, 75°F"
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")

@tool
def calculate(expression: str) -> str:
    """Safely evaluate a mathematical expression."""
    try:
        allowed_chars = set("0123456789+-*/(). ")
        if all(c in allowed_chars for c in expression):
            result = eval(expression)
            return f"Result: {result}"
        return "Invalid expression"
    except Exception as e:
        return f"Error: {str(e)}"

tools = [get_weather, calculate]
tool_node = ToolNode(tools)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

def should_continue(state: AgentState) -> Literal["tools", "end"]:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "end"

def call_model(state: AgentState):
    system_msg = SystemMessage(content="You are a helpful assistant with access to weather and calculator tools.")
    messages = [system_msg] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

def create_checkpointer():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return SqliteSaver(conn)

def build_graph(checkpointer):
    workflow = StateGraph(AgentState)

    # Define the two nodes:
    #   agent: calls the LLM, which may respond with tool calls.
    #   tools: executes any requested tools from the last agent message.
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    # Entry point: start in the agent node.
    workflow.set_entry_point("agent")

    # Conditional transition after the agent:
    # - If the last model message contains tool_calls, route to "tools".
    # - Otherwise, the conversation is complete and we end.
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})

    # After tools run, always return to the agent so the model can
    # process the tool results and generate a final answer.
    workflow.add_edge("tools", "agent")

    # Optional: visualize the graph (requires graphviz/pydot).
    # e.g. graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
    return workflow.compile(checkpointer=checkpointer)

def run_conversation(thread_id: str, user_id: str, user_input: str, graph):
    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
    state = {"messages": [HumanMessage(content=user_input)], "user_id": user_id, "thread_id": thread_id}
    result = graph.invoke(state, config=config)
    return result["messages"][-1].content

def get_conversation_history(thread_id: str, user_id: str, graph):
    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
    try:
        state = graph.get_state(config)
        if state and state.values:
            return state.values.get("messages", [])
    except Exception:
        pass
    return []

def list_threads(user_id: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT thread_id, MAX(created_at) as last_updated
        FROM checkpoints
        WHERE thread_id LIKE ?
        GROUP BY thread_id
        ORDER BY last_updated DESC
    """, (f"%{user_id}%",))
    threads = cursor.fetchall()
    conn.close()
    return threads

def main():
    print("=" * 60)
    print("LangGraph SQLite Checkpointing Demo")
    print("=" * 60)

    checkpointer = create_checkpointer()
    graph = build_graph(checkpointer)

    # Optional graph visualization (uncomment if you want to generate a diagram):
    # graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
    # graph.get_graph().print_ascii()

    user_id = "user_123"
    thread_id = "thread_1"

    # --- Session 1: First conversation in this thread ---
    print(f"\nUser: {user_id}")
    print(f"Thread: {thread_id}")
    print("\n--- First message ---")
    response1 = run_conversation(thread_id, user_id, "What's the weather in New York?", graph)
    print(f"Assistant: {response1}")

    print("\n--- Second message (should trigger tools) ---")
    response2 = run_conversation(thread_id, user_id, "And what's 15 * 14?")
    print(f"Assistant: {response2}")

    print("\n--- Checking checkpoint state ---")
    state = get_conversation_history(thread_id, user_id, graph)
    print(f"Messages in thread: {len(state)}")
    for msg in state:
        role = "Human" if isinstance(msg, HumanMessage) else "AI"
        content = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        print(f"  [{role}] {content}")

    print("\n--- New session, same thread (memory persists) ---")
    response3 = run_conversation(thread_id, user_id, "What was the weather in New York?")
    print(f"Assistant: {response3}")

    print("\n--- New thread (separate memory) ---")
    thread2 = "thread_2"
    response4 = run_conversation(thread2, user_id, "What is the weather in Paris?")
    print(f"Assistant: {response4}")

    print("\n--- Listing all threads for user ---")
    threads = list_threads(user_id)
    for tid, last_updated in threads:
        print(f"  Thread: {tid} | Last updated: {last_updated}")

if __name__ == "__main__":
    main()
