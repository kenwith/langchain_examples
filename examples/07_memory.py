"""This module demonstrates the different types of memory available in LangChain
and provides guidance on how to choose between them.

Memory in LangChain is used to store and retrieve conversation history so that
LLM-powered applications can maintain context across multiple interactions.
There are several built-in memory types, each with its own tradeoffs in terms of
information retention, token usage, and complexity.

Memory Types:
1. ConversationBufferMemory
   - Stores the entire conversation history in a buffer.
   - Provides full context, but can become expensive and exceed token limits as
     conversations grow.
   - Best for short conversations or when complete history is essential.

2. ConversationBufferWindowMemory
   - Keeps only the most recent `k` exchanges in the buffer.
   - Limits token usage by forgetting older messages.
   - Best for applications where recent context matters more than complete
     history, such as chatbots with a conversational window.

3. ConversationSummaryMemory
   - Maintains a running summary of the conversation instead of raw messages.
   - Reduces token usage significantly and never forgets the distilled context.
   - Best for long-running conversations where summarizing earlier parts is
     acceptable and the full verbatim history is not necessary.

4. ConversationSummaryBufferMemory
   - Combines a token-based buffer with a summary. It keeps recent messages in
     full, and summarizes older messages once the token threshold is exceeded.
   - Provides a good balance between preserving recent detail and controlling
     cost.
   - Best for long conversations where you want recent exchanges in full but
     also want historical context through summaries.

How to choose:
- If conversations are short and you need verbatim recall: ConversationBufferMemory.
- If you need to bound token usage and only care about recent context:
  ConversationBufferWindowMemory.
- If conversations are long and token costs are a concern, and exact wording is
  not critical: ConversationSummaryMemory.
- If you want a mix of recent messages and summarized older context:
  ConversationSummaryBufferMemory.

This module illustrates each memory type with a conversation chain.
"""

import os

from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryBufferMemory,
    ConversationSummaryMemory,
)

# Uncomment and set your API key if not already configured in the environment:
# os.environ["OPENAI_API_KEY"] = "your-api-key"


def create_chain(memory):
    """Create a conversation chain using the given memory."""
    llm = OpenAI(temperature=0)
    return ConversationChain(llm=llm, memory=memory, verbose=True)


def demonstrate_buffer_memory():
    print("\n" + "=" * 60)
    print("Demonstrating ConversationBufferMemory")
    print("=" * 60)
    memory = ConversationBufferMemory()
    chain = create_chain(memory)
    chain.predict(input="Hi, my name is Alice.")
    chain.predict(input="What is my name?")


def demonstrate_window_memory():
    print("\n" + "=" * 60)
    print("Demonstrating ConversationBufferWindowMemory (k=1)")
    print("=" * 60)
    memory = ConversationBufferWindowMemory(k=1)
    chain = create_chain(memory)
    chain.predict(input="Hi, I like pizza.")
    chain.predict(input="What do I like?")
    chain.predict(input="What was my first message?")
    # The last question will not recall the first message because k=1.


def demonstrate_summary_memory():
    print("\n" + "=" * 60)
    print("Demonstrating ConversationSummaryMemory")
    print("=" * 60)
    memory = ConversationSummaryMemory(llm=OpenAI(temperature=0))
    chain = create_chain(memory)
    chain.predict(input="Hello, I am Bob.")
    chain.predict(input="What is my name?")


def demonstrate_summary_buffer_memory():
    print("\n" + "=" * 60)
    print("Demonstrating ConversationSummaryBufferMemory")
    print("=" * 60)
    memory = ConversationSummaryBufferMemory(
        llm=OpenAI(temperature=0), max_token_limit=100
    )
    chain = create_chain(memory)
    chain.predict(
        input="I am learning about LangChain. It provides modules to build "
        "LLM applications. Memory is one of the important components."
    )
    chain.predict(input="What is the summary of what I said?")


def main():
    demonstrate_buffer_memory()
    demonstrate_window_memory()
    demonstrate_summary_memory()
    demonstrate_summary_buffer_memory()


if __name__ == "__main__":
    main()
