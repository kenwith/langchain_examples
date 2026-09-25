"""
Example: Agent with Memory using LangGraph

This example demonstrates how to build a LangGraph agent that maintains
conversation memory using a checkpoint saver. The agent can remember
information from previous interactions within the same session.
"""

import os
from typing import List, Optional

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

# -------------------------------
# Tools
# -------------------------------

@tool
def get_current_time() -> str:
    """Returns the current time in UTC."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

# -------------------------------
# Agent Creation
# -------------------------------

def create_agent_with_memory(model_name: str, model_provider: str, system_prompt: Optional[str] = None):
    """Create a LangGraph agent with conversation memory."""
    # Initialize the chat model provider-agnostically
    model = init_chat_model(model_name, model_provider=model_provider)
    
    # Set up in-memory checkpointing to store conversation state
    memory = InMemorySaver()
    
    # Create the agent with tools and checkpointing
    tools = [get_current_time]
    agent = create_react_agent(
        model,
        tools,
        prompt=system_prompt,  # optional system prompt
        checkpointer=memory,   # enables memory across turns
    )
    return agent

# -------------------------------
# Conversation Runner
# -------------------------------

def run_conversation(agent, config: dict, messages: List[str]) -> None:
    """Run a series of user messages through the agent, printing responses."""
    all_messages = []  # collect all messages for summary
    for user_input in messages:
        print(f"User: {user_input}")
        response = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,  # config contains thread_id for memory
        )
        # Extract the AI response (last message)
        ai_message = response["messages"][-1]
        print(f"Assistant: {ai_message.content}\n")
        # Store for summary
        all_messages.append({"role": "user", "content": user_input})
        all_messages.append({"role": "assistant", "content": ai_message.content})
    
    # Print compact summary after the conversation
    print_conversation_summary(all_messages)

def print_conversation_summary(messages: List[dict]) -> None:
    """Print a compact summary of the conversation."""
    print("\n" + "="*50)
    print("COMPACT CONVERSATION SUMMARY")
    print("="*50)
    for i, msg in enumerate(messages, 1):
        role = msg["role"].capitalize()
        content = msg["content"]
        # Truncate long messages for compactness
        if len(content) > 80:
            content = content[:77] + "..."
        print(f"{i:2d}. {role}: {content}")
    print("="*50)

# -------------------------------
# Main Demo
# -------------------------------

if __name__ == "__main__":
    # Provider-agnostic configuration via environment variables
    # Default to OpenAI, but can be changed by setting MODEL_PROVIDER and MODEL_NAME
    model_provider = os.getenv("MODEL_PROVIDER", "openai")
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    
    # Check for API key based on provider (example: OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
    # This is a simplified check; you may need to set the appropriate key for your provider.
    if not os.getenv("OPENAI_API_KEY") and model_provider == "openai":
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    # For other providers, you might want to add similar checks here.
    
    # Create agent with memory
    agent = create_agent_with_memory(model_name, model_provider)
    
    # Configuration for a single conversation thread
    config = {"configurable": {"thread_id": "example-thread-1"}}
    
    # Simulate a conversation where memory matters
    messages = [
        "Hi, my name is Alice.",
        "What is my name?",
        "Can you tell me the current time?",
        "What did I tell you earlier?",
    ]
    
    run_conversation(agent, config, messages)
