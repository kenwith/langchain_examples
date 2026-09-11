"""
Example: Using trim_conversation to cap stored messages and prevent context overflow.
Demonstrates both in-memory and persistent (SQLite) chat message history.
"""

import os

from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import SQLiteChatMessageHistory


def trim_conversation(memory, max_messages=10):
    """Trim the stored chat history to the last `max_messages` messages.

    Works with both in-memory and persistent chat histories by clearing
    the history and re-adding only the retained messages.

    Args:
        memory: A LangChain memory object with a `chat_memory.messages` list.
        max_messages: Maximum number of messages to keep.

    Returns:
        The same memory object, with the message list truncated.
    """
    messages = memory.chat_memory.messages
    if len(messages) > max_messages:
        retained = messages[-max_messages:]
        memory.chat_memory.clear()
        for msg in retained:
            memory.chat_memory.add_message(msg)
    return memory


def create_memory(persist: bool):
    """Create a ConversationBufferMemory, optionally backed by SQLite.

    Args:
        persist: If True, use a SQLite-backed chat message history.
                 If False, use the default in-memory history.

    Returns:
        A ConversationBufferMemory instance.
    """
    if persist:
        # SQLite-backed history persists across script runs.
        # Trade-off: requires disk I/O and a database file, but survives restarts.
        chat_history = SQLiteChatMessageHistory(
            session_id="demo_session",
            connection_string="sqlite:///memory.db",
        )
        return ConversationBufferMemory(
            chat_memory=chat_history,
            return_messages=True,
        )
    else:
        # In-memory history is fast and ephemeral.
        # Trade-off: data is lost when the process ends.
        return ConversationBufferMemory(return_messages=True)


def simulate_conversation(memory, turns=15):
    """Add simulated user/AI message pairs to the memory."""
    for i in range(turns):
        memory.chat_memory.add_user_message(f"Hello, I am message {i}")
        memory.chat_memory.add_ai_message(f"Hi, I am response {i}")


def main():
    """Run a simple demonstration of trimming conversation memory."""
    # Use an environment variable for the API key; never hardcode secrets.
    # This demo doesn't call an LLM, but the check is included as a reminder.
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable.")
        return

    print("=== In-memory memory ===")
    memory = create_memory(persist=False)
    simulate_conversation(memory)

    print(f"Before trimming: {len(memory.chat_memory.messages)} messages")
    trim_conversation(memory, max_messages=6)
    print(f"After trimming: {len(memory.chat_memory.messages)} messages")
    for msg in memory.chat_memory.messages:
        print(f"{msg.type}: {msg.content}")

    print("\n=== Persistent (SQLite) memory ===")
    memory = create_memory(persist=True)
    simulate_conversation(memory)

    print(f"Before trimming: {len(memory.chat_memory.messages)} messages")
    trim_conversation(memory, max_messages=6)
    print(f"After trimming: {len(memory.chat_memory.messages)} messages")
    for msg in memory.chat_memory.messages:
        print(f"{msg.type}: {msg.content}")

    print("\nNote: The SQLite-backed memory persists across runs.")
    print("Run this script again to see the previous session's trimmed history.")


if __name__ == "__main__":
    main()
