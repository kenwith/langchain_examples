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
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    workflow.add_edge("tools", "agent")
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
    
    user_id = "user_123"
    thread_id = "thread_abc"
    
    print(f"\nUser: {user_id}")
    print(f"Thread: {thread_id}")
    print("-" * 40)
    
    print("\n--- First Conversation Session ---")
    response1 = run_conversation(thread_id, user_id, "Hi! What's the weather in New York?", graph)
    print(f"Assistant: {response1}")
    
    response2 = run_conversation(thread_id, user_id, "Thanks! Can you calculate 25 * 4 + 10?", graph)
    print(f"Assistant: {response2}")
    
    print("\n--- Simulating New Session (Same Thread) ---")
    print("Loading conversation history...")
    history = get_conversation_history(thread_id, user_id, graph)
    print(f"Found {len(history)} messages in history")
    for msg in history:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        content = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        print(f"  [{role}] {content}")
    
    print("\n--- Continuing Conversation ---")
    response3 = run_conversation(thread_id, user_id, "What was the weather I asked about earlier?", graph)
    print(f"Assistant: {response3}")
    
    print("\n--- Starting New Thread ---")
    new_thread_id = "thread_xyz"
    response4 = run_conversation(new_thread_id, user_id, "Hello! What's the weather in London?", graph)
    print(f"Assistant: {response4}")
    
    print("\n--- Thread Listing ---")
    threads = list_threads(user_id)
    print(f"Threads for {user_id}:")
    for thread_id_db, last_updated in threads:
        print(f"  - {thread_id_db} (last updated: {last_updated})")
    
    print("\n--- Checkpoint Inspection ---")
    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
    state = graph.get_state(config)
    if state:
        print(f"Current state keys: {list(state.values.keys()) if state.values else 'None'}")
        print(f"Next nodes: {state.next}")
        print(f"Checkpoint metadata: {state.metadata}")
    
    print("\n" + "=" * 60)
    print("Demo complete! Checkpoints saved to:", DB_PATH)
    print("=" * 60)

if __name__ == "__main__":
    main()
