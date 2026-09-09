"""
Memory & Conversation History Example

Demonstrates: ConversationBufferMemory, ConversationSummaryMemory, 
ConversationSummaryBufferMemory, vector store memory, custom memory with LangGraph
Provider-agnostic using init_chat_model
"""
import os
from typing import Annotated
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryBufferMemory,
)
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

load_dotenv()


def get_model():
    model_name = os.getenv("LANGCHAIN_MODEL", "openai/gpt-4o-mini")
    return init_chat_model(model_name)


# =============================================================================
# Helper: Print conversation history clearly
# =============================================================================

def print_history(messages, max_chars=100):
    """Print a list of messages in a readable format, truncating long content."""
    print(f"  ({len(messages)} messages)")
    for i, msg in enumerate(messages):
        content = msg.content
        if len(content) > max_chars:
            content = content[:max_chars] + "..."
        print(f"    [{i}] {msg.type}: {content}")


# =============================================================================
# 1. ConversationBufferMemory (Full History with trimming)
# =============================================================================

def buffer_memory_example():
    """Basic buffer memory - stores all messages, but trims to last N to prevent unbounded growth"""
    print("=== ConversationBufferMemory (with trimming) ===")

    model = get_model()
    memory = ConversationBufferMemory(return_messages=True)
    MAX_MESSAGES = 4  # Keep last 2 exchanges (4 messages)

    # Simulate conversation
    conversation = [
        ("user", "Hi, I'm Alice"),
        ("assistant", "Hello Alice! How can I help?"),
        ("user", "What's my name?"),
    ]

    for role, content in conversation:
        if role == "user":
            memory.chat_memory.add_user_message(content)
        else:
            memory.chat_memory.add_ai_message(content)

    # Trim to last MAX_MESSAGES
    if len(memory.chat_memory.messages) > MAX_MESSAGES:
        memory.chat_memory.messages = memory.chat_memory.messages[-MAX_MESSAGES:]

    # Create chain with memory
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("user", "{input}"),
    ])

    chain = prompt | model | StrOutputParser()

    # Load memory variables
    memory_vars = memory.load_memory_variables({})
    print("Memory contents:")
    print_history(memory_vars["history"])

    # Continue conversation
    result = chain.invoke({
        "input": "What did I say my name was?",
        "history": memory_vars["history"]
    })
    print(f"Response: {result}")

    # Update memory (and trim again)
    memory.chat_memory.add_user_message("What did I say my name was?")
    memory.chat_memory.add_ai_message(result)
    if len(memory.chat_memory.messages) > MAX_MESSAGES:
        memory.chat_memory.messages = memory.chat_memory.messages[-MAX_MESSAGES:]

    print("\nAfter adding response, memory now contains:")
    memory_vars = memory.load_memory_variables({})
    print_history(memory_vars["history"])


# =============================================================================
# 2. ConversationBufferWindowMemory (Last N messages)
# =============================================================================

def window_memory_example():
    """Window memory - keeps only last k messages"""
    print("\n=== ConversationBufferWindowMemory (k=2) ===")

    model = get_model()
    memory = ConversationBufferWindowMemory(k=2, return_messages=True)

    # Add many messages
    for i in range(5):
        memory.chat_memory.add_user_message(f"Message {i}")
        memory.chat_memory.add_ai_message(f"Response {i}")

    memory_vars = memory.load_memory_variables({})
    print("Stored messages (should be last 2 pairs = 4 messages):")
    print_history(memory_vars["history"])


# =============================================================================
# 3. ConversationSummaryMemory (Summarized History)
# =============================================================================

def summary_memory_example():
    """Summary memory - keeps summary instead of full history"""
    print("\n=== ConversationSummaryMemory ===")

    model = get_model()
    memory = ConversationSummaryMemory(llm=model, return_messages=True)

    # Add conversation
    conversation = [
        ("user", "I'm planning a trip to Japan"),
        ("assistant", "Great! Japan is wonderful. When are you going?"),
        ("user", "Next spring, during cherry blossom season"),
        ("assistant", "Perfect timing! Kyoto and Tokyo are beautiful then."),
        ("user", "Any specific recommendations?"),
    ]

    for role, content in conversation:
        if role == "user":
            memory.chat_memory.add_user_message(content)
        else:
            memory.chat_memory.add_ai_message(content)

    memory_vars = memory.load_memory_variables({})
    print("Summary memory:")
    for msg in memory_vars["history"]:
        content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
        print(f"  {msg.type}: {content}")


# =============================================================================
# 4. ConversationSummaryBufferMemory (Hybrid: Summary + Recent Buffer)
# =============================================================================

def summary_buffer_memory_example():
    """SummaryBufferMemory - condenses old messages while keeping recent history intact"""
    print("\n=== ConversationSummaryBufferMemory ===")

    model = get_model()
    # max_token_limit: total tokens for buffer + summary
    # When buffer exceeds limit, oldest messages are summarized
    memory = ConversationSummaryBufferMemory(
        llm=model,
        max_token_limit=300,  # Small limit for demo
        return_messages=True,
    )

    # Simulate a longer conversation
    conversation = [
        ("user", "Hi, I'm planning a trip to Japan next spring"),
        ("assistant", "That sounds wonderful! Spring is beautiful in Japan with cherry blossoms."),
        ("user", "I'm particularly interested in Kyoto and Tokyo"),
        ("assistant", "Excellent choices! Kyoto has temples and gardens, Tokyo has modern attractions."),
        ("user", "What's the best time to see cherry blossoms in Kyoto?"),
        ("assistant", "Typically late March to early April. Philosopher's Path and Maruyama Park are great spots."),
        ("user", "Any food recommendations in Tokyo?"),
        ("assistant", "Try sushi at Tsukiji, ramen in Shinjuku, and street food in Asakusa."),
        ("user", "I also want to visit Osaka for a day trip"),
        ("assistant", "Osaka is great for food! Try takoyaki and okonomiyaki in Dotonbori."),
        ("user", "What about transportation between cities?"),
        ("assistant", "The Shinkansen (bullet train) is fastest. Japan Rail Pass can save money."),
        ("user", "Do I need to book Shinkansen tickets in advance?"),
        ("assistant", "Not required but recommended for peak seasons. Can reserve at stations."),
        ("user", "What's the weather like in spring?"),
        ("assistant", "Mild, 10-20°C. Pack layers and a light jacket for evenings."),
        ("user", "Any cultural etiquette I should know?"),
        ("assistant", "Remove shoes indoors, don't tip, be quiet on trains, respect queues."),
    ]

    print("Adding conversation history...")
    for role, content in conversation:
        if role == "user":
            memory.chat_memory.add_user_message(content)
        else:
            memory.chat_memory.add_ai_message(content)

    memory_vars = memory.load_memory_variables({})
    print(f"\nMemory contents ({len(memory_vars['history'])} messages):")
    print_history(memory_vars["history"])

    # Show the buffer vs summary breakdown
    print(f"\n--- Memory Structure ---")
    print(f"Buffer (recent messages kept verbatim): {len(memory.buffer)} messages")
    print(f"Summary (condensed older messages): {memory.moving_summary_buffer[:200]}..." if memory.moving_summary_buffer else "Summary: (empty)")

    # Continue conversation - should have context from both summary and buffer
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a travel assistant. Use the conversation history to answer."),
        MessagesPlaceholder(variable_name="history"),
        ("user", "{input}"),
    ])

    chain = prompt | model | StrOutputParser()

    print("\n--- Continuing conversation ---")
    follow_up = "What was the first city I mentioned wanting to visit?"
    result = chain.invoke({
        "input": follow_up,
        "history": memory_vars["history"]
    })
    print(f"User: {follow_up}")
    print(f"Bot: {result}")

    # Add to memory
    memory.chat_memory.add_user_message(follow_up)
    memory.chat_memory.add_ai_message(result)

    # Check memory after adding
    memory_vars = memory.load_memory_variables({})
    print(f"\nMemory after follow-up ({len(memory_vars['history'])} messages):")
    print_history(memory_vars["history"])


# =============================================================================
# 5. RunnableWithMessageHistory (LCEL Pattern with trimming)
# =============================================================================

def runnable_with_history():
    """Using RunnableWithMessageHistory for LCEL chains, with session history trimming"""
    print("\n=== RunnableWithMessageHistory ===")

    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langchain_community.chat_message_histories import ChatMessageHistory

    model = get_model()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Remember user details."),
        MessagesPlaceholder(variable_name="history"),
        ("user", "{input}"),
    ])

    chain = prompt | model | StrOutputParser()

    # In-memory store for session histories
    store = {}
    MAX_SESSION_MESSAGES = 6  # Keep last 3 exchanges

    def get_session_history(session_id: str):
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        # Trim to last MAX_SESSION_MESSAGES
        if len(store[session_id].messages) > MAX_SESSION_MESSAGES:
            store[session_id].messages = store[session_id].messages[-MAX_SESSION_MESSAGES:]
        return store[session_id]

    with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    config = {"configurable": {"session_id": "user-123"}}

    # First message
    result1 = with_history.invoke({"input": "My favorite color is blue"}, config=config)
    print(f"Response 1: {result1}")

    # Second message - should remember
    result2 = with_history.invoke({"input": "What's my favorite color?"}, config=config)
    print(f"Response 2: {result2}")

    # Different session - should not remember
    result3 = with_history.invoke(
        {"input": "What's my favorite color?"},
        config={"configurable": {"session_id": "user-456"}}
    )
    print(f"Response 3 (new session): {result3}")

    # Show history for session 1
    print("\nSession 1 history:")
    print_history(store["user-123"].messages)


# =============================================================================
# 6. LangGraph State-Based Memory (with trimming)
# =============================================================================

def langgraph_memory():
    """Memory using LangGraph state and checkpointer, with message trimming"""
    print("\n=== LangGraph State Memory ===")

    model = get_model()
    memory = MemorySaver()
    MAX_STATE_MESSAGES = 6  # Keep last 3 exchanges

    class ChatState(BaseModel):
        messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
        user_name: str = ""

    def chat_node(state: ChatState):
        # Extract user name if mentioned
        last_msg = state.messages[-1].content if state.messages else ""
        if "my name is" in last_msg.lower() and not state.user_name:
            # Simple extraction
            name = last_msg.split("my name is")[-1].strip().split()[0].rstrip(".")
            state.user_name = name.capitalize()

        # Trim messages to last MAX_STATE_MESSAGES
        if len(state.messages) > MAX_STATE_MESSAGES:
            state.messages = state.messages[-MAX_STATE_MESSAGES:]

        # Build prompt with context
        system_msg = SystemMessage(
            content=f"You are a helpful assistant. User's name: {state.user_name or 'Unknown'}"
        )
        messages = [system_msg] + state.messages
        result = model.invoke(messages)
        return {"messages": [result], "user_name": state.user_name}

    workflow = StateGraph(ChatState)
    workflow.add_node("chat", chat_node)
    workflow.add_edge(START, "chat")
    workflow.add_edge("chat", END)

    app = workflow.compile(checkpointer=memory)

    config = {"configurable": {"thread_id": "session-abc"}}

    # Conversation
    for msg in [
        "Hi, my name is Bob",
        "What's my name?",
        "I like pizza",
        "What do I like?",
    ]:
        print(f"\nUser: {msg}")
        result = app.invoke({"messages": [HumanMessage(content=msg)]}, config=config)
        print(f"Bot: {result['messages'][-1].content}")

    # Show final state
    final_state = app.get_state(config)
    print("\nFinal state messages:")
    print_history(final_state.values["messages"])


# =============================================================================
# 7. Vector Store Memory (Semantic Retrieval)
# =============================================================================

def vector_memory_example():
    """Memory using vector store for semantic retrieval"""
    print("\n=== Vector Store Memory ===")

    model = get_model()

    try:
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_core.documents import Document

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Store conversation as documents
        conversations = [
            Document(page_content="User likes pizza and pasta", metadata={"type": "preference"}),
            Document(page_content="User works as a software engineer", metadata={"type": "work"}),
            Document(page_content="User is learning Japanese", metadata={"type": "learning"}),
            Document(page_content="User visited Japan in 2023", metadata={"type": "travel"}),
        ]

        vector_store = FAISS.from_documents(conversations, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})

        # Retrieve relevant context
        query = "What does the user like to eat?"
        docs = retriever.invoke(query)
        context = "\n".join(d.page_content for d in docs)

        # Generate response with context
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer based on the context:\n{context}"),
            ("user", "{question}"),
        ])

        chain = prompt | model | StrOutputParser()
        result = chain.invoke({"question": query, "context": context})
        print(f"Q: {query}")
        print(f"A: {result}")

    except ImportError:
        print("FAISS not installed - skipping vector memory example")


# =============================================================================
# 8. Custom Memory Class (with improved trimming and printing)
# =============================================================================

class CustomMemory:
    """Custom memory with specific behavior"""

    def __init__(self, max_tokens: int = 2000):
        self.messages: list[BaseMessage] = []
        self.max_tokens = max_tokens
        self.model = get_model()

    def add_message(self, message: BaseMessage):
        self.messages.append(message)
        self._trim()

    def _trim(self):
        """Trim old messages if exceeding token budget"""
        # Simple approximation: 4 chars per token
        total_chars = sum(len(m.content) for m in self.messages)
        while total_chars > self.max_tokens * 4 and len(self.messages) > 1:
            # Remove oldest non-system message
            for i, msg in enumerate(self.messages):
                if msg.type != "system":
                    removed = self.messages.pop(i)
                    total_chars -= len(removed.content)
                    break

    def get_messages(self) -> list[BaseMessage]:
        return self.messages

    def clear(self):
        self.messages = []


def custom_memory_example():
    """Custom memory implementation"""
    print("\n=== Custom Memory ===")

    memory = CustomMemory(max_tokens=100)  # Very small for demo

    # Add messages
    memory.add_message(SystemMessage(content="You are helpful."))
    for i in range(10):
        memory.add_message(HumanMessage(content=f"Message {i}"))
        memory.add_message(AIMessage(content=f"Response {i}"))

    print(f"Stored {len(memory.get_messages())} messages (trimmed to fit):")
    print_history(memory.get_messages())


if __name__ == "__main__":
    buffer_memory_example()
    window_memory_example()
    summary_memory_example()
    summary_buffer_memory_example()
    runnable_with_history()
    langgraph_memory()
    vector_memory_example()
    custom_memory_example()
    print("\nAll memory examples completed!")
