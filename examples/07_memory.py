import os

from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


class ConversationSession:
    """A session-scoped conversation using ConversationBufferMemory.

    ConversationBufferMemory retains the full chat history across turns, so the
    model can refer back to earlier messages. This class provides a simple
    ask/response interface and a clear_history helper.
    """

    def __init__(self, model="text-davinci-003", temperature=0.7):
        self.llm = OpenAI(
            temperature=temperature,
            model_name=model,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        # ConversationBufferMemory stores the complete message history in memory.
        self.memory = ConversationBufferMemory()
        self.chain = ConversationChain(llm=self.llm, memory=self.memory, verbose=True)

    def ask(self, prompt: str) -> str:
        """Send a prompt to the conversation and return the AI response."""
        return self.chain.run(input=prompt)

    def clear_history(self) -> None:
        """Clear the conversation memory for this session."""
        self.memory.clear()

    def format_history(self) -> str:
        """Return a formatted string of the conversation history."""
        memory_vars = self.memory.load_memory_variables({})
        history = memory_vars.get("history", "")
        if not history:
            return "No conversation history yet."
        return f"Conversation history:\n{history}"


class ConversationSummarySession:
    """A session-scoped conversation using ConversationSummaryMemory.

    ConversationSummaryMemory keeps a running summary of the conversation instead
    of storing the full message history. This is useful for long conversations
    where you want to reduce token usage while retaining the key context.

    Usage:
        session = ConversationSummarySession()
        session.ask("My favorite color is blue.")
        session.ask("What is my favorite color?")
        session.format_history()
        session.clear_history()
    """

    def __init__(self, model="text-davinci-003", temperature=0.7):
        self.llm = OpenAI(
            temperature=temperature,
            model_name=model,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.memory = ConversationSummaryMemory(llm=self.llm)
        self.chain = ConversationChain(llm=self.llm, memory=self.memory, verbose=True)

    def ask(self, prompt: str) -> str:
        """Send a prompt to the conversation and return the AI response."""
        return self.chain.run(input=prompt)

    def clear_history(self) -> None:
        """Clear the conversation memory for this session."""
        self.memory.clear()

    def format_history(self) -> str:
        """Return a formatted string of the conversation summary."""
        memory_vars = self.memory.load_memory_variables({})
        history = memory_vars.get("history", "")
        if not history:
            return "No conversation history yet."
        return f"Conversation summary:\n{history}"


class RunnableConversationSession:
    """A session-scoped conversation using RunnableWithMessageHistory and an in-memory store."""

    def __init__(self, model="gpt-3.5-turbo", temperature=0.7):
        self.llm = ChatOpenAI(
            temperature=temperature,
            model_name=model,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant."),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )
        self.chain = self.prompt | self.llm
        self.store = {}
        self.history = RunnableWithMessageHistory(
            self.chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

    def get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]

    def ask(self, prompt: str, session_id: str = "default") -> str:
        """Send a prompt to the conversation and return the AI response."""
        response = self.history.invoke(
            {"input": prompt},
            config={"configurable": {"session_id": session_id}},
        )
        return response.content

    def clear_history(self, session_id: str = "default") -> None:
        """Clear the conversation memory for a session."""
        self.store.pop(session_id, None)

    def format_history(self, session_id: str = "default") -> str:
        """Return a formatted string of the conversation history."""
        history = self.store.get(session_id)
        if not history or not history.messages:
            return "No conversation history yet."
        lines = []
        for message in history.messages:
            if message.type == "human":
                lines.append(f"Human: {message.content}")
            elif message.type == "ai":
                lines.append(f"AI: {message.content}")
        return "Conversation history:\n" + "\n".join(lines)


def main() -> None:
    # ConversationBufferMemory retains the full chat history across turns.
    print("\n--- ConversationBufferMemory Example ---\n")

    # Each ConversationSession has its own isolated memory.
    session = ConversationSession()

    print("AI:", session.ask("My favorite color is blue."))
    print("AI:", session.ask("What is my favorite color?"))

    # Show the formatted history before clearing.
    print("\n" + session.format_history() + "\n")

    # Clear the session history to start a fresh multi-turn demonstration.
    session.clear_history()
    print("\nHistory cleared. The AI now remembers nothing from the previous turns.\n")

    print("AI:", session.ask("What is my favorite color?"))

    # ConversationSummaryMemory example with a running summary.
    print("\n--- ConversationSummaryMemory Example ---\n")
    summary_session = ConversationSummarySession()

    print("AI:", summary_session.ask("My favorite color is blue."))
    print("AI:", summary_session.ask("What is my favorite color?"))

    # Show the formatted summary before clearing.
    print("\n" + summary_session.format_history() + "\n")

    # Clear the session history to start a fresh multi-turn demonstration.
    summary_session.clear_history()
    print("\nHistory cleared. The AI now remembers nothing from the previous turns.\n")

    print("AI:", summary_session.ask("What is my favorite color?"))

    # RunnableWithMessageHistory example with a simple in-memory chat history store.
    print("\n--- RunnableWithMessageHistory Example ---\n")
    runnable_session = RunnableConversationSession()

    print("AI:", runnable_session.ask("My favorite color is blue.", session_id="user-1"))
    print("AI:", runnable_session.ask("What is my favorite color?", session_id="user-1"))

    # Show the formatted history before clearing.
    print("\n" + runnable_session.format_history(session_id="user-1") + "\n")

    # Clear the session history for user-1.
    runnable_session.clear_history(session_id="user-1")
    print("\nHistory cleared for user-1. The AI now remembers nothing from the previous turns.\n")

    print("AI:", runnable_session.ask("What is my favorite color?", session_id="user-1"))


if __name__ == "__main__":
    main()
