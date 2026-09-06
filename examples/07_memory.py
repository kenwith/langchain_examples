"""
Memory & Conversation History Example

Demonstrates: ConversationBufferMemory, ConversationSummaryMemory, 
vector store memory, custom memory with LangGraph
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
# 1. ConversationBufferMemory (Full History)
# =============================================================================

def buffer_memory_example():
    """Basic buffer memory - stores all messages"""
    print("=== ConversationBufferMemory ===")

    model = get_model()
    memory = ConversationBufferMemory(return_messages=True)

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

    # Create chain with memory
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("user", "{input}"),
    ])

    chain = prompt | model | StrOutputParser()

    # Load memory variables
    memory_vars = memory.load_memory_variables({})
    print(f"Memory: {memory_vars}")

    # Continue conversation
    result = chain.invoke({
        "input": "What did I say my name was?",
        "history": memory_vars["history"]
    })
    print(f"Response: {result}")

    # Update memory
    memory.chat_memory.add_user_message("What did I say my name was?")
    memory.chat_memory.add_ai_message(result)


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
    print(f"Stored messages (should be last 2 pairs = 4 messages):")
    for msg in memory_vars["history"]:
        print(f"  {msg.type}: {msg.content}")


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
    print(f"Summary memory:")
    for msg in memory_vars["history"]:
        print(f"  {msg.type}: {msg.content[:100]}...")


# =============================================================================
# 4. RunnableWithMessageHistory (LCEL Pattern)
# =============================================================================

def runnable_with_history():
    """Using RunnableWithMessageHistory for LCEL chains"""
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

    def get_session_history(session_id: str):
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
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


# =============================================================================
# 5. LangGraph State-Based Memory
# =============================================================================

def langgraph_memory():
    """Memory using LangGraph state and checkpointer"""
    print("\n=== LangGraph State Memory ===")

    model = get_model()
    memory = MemorySaver()

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


# =============================================================================
# 6. Vector Store Memory (Semantic Retrieval)
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
# 7. Custom Memory Class
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

    print(f"Stored {len(memory.get_messages())} messages (trimmed to fit)")
    for msg in memory.get_messages():
        print(f"  {msg.type}: {msg.content[:30]}...")


if __name__ == "__main__":
    buffer_memory_example()
    window_memory_example()
    summary_memory_example()
    runnable_with_history()
    langgraph_memory()
    vector_memory_example()
    custom_memory_example()
    print("\nAll memory examples completed!")