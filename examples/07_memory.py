"""
Example: Using trim_conversation to cap stored messages and prevent context overflow.
"""
import os

from langchain.memory import ConversationBufferMemory


def trim_conversation(memory, max_messages=10):
    """Trim the stored chat history to the last `max_messages` messages.

    Args:
        memory: A LangChain memory object with a `chat_memory.messages` list.
        max_messages: Maximum number of messages to keep.

    Returns:
        The same memory object, with the message list truncated.
    """
    messages = memory.chat_memory.messages
    if len(messages) > max_messages:
        memory.chat_memory.messages = messages[-max_messages:]
    return memory


def main():
    """Run a simple demonstration of trimming conversation memory."""
    # Use an environment variable for the API key; never hardcode secrets.
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable.")
        return

    memory = ConversationBufferMemory(return_messages=True)

    # Simulate a long conversation (no actual LLM calls needed for the demo).
    for i in range(15):
        memory.chat_memory.add_user_message(f"Hello, I am message {i}")
        memory.chat_memory.add_ai_message(f"Hi, I am response {i}")

    print(f"Before trimming: {len(memory.chat_memory.messages)} messages")

    trim_conversation(memory, max_messages=6)

    print(f"After trimming: {len(memory.chat_memory.messages)} messages")
    for msg in memory.chat_memory.messages:
        print(f"{msg.type}: {msg.content}")


if __name__ == "__main__":
    main()
