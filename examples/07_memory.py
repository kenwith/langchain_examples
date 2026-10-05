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

    Memory configuration:
    - memory_key: the variable name passed into the chain's prompt template.
      ConversationChain expects this to be "history" by default.
    - input_key: the input field that contains the user's latest message.
    - return_messages: set to True to receive history as a list of chat messages
      (useful with chat models); set to False to receive a formatted string.
    - human_prefix/ai_prefix: labels used when rendering string history as
      "Human: ..." and "AI: ...". Override them if you need different labels.
    """

    def __init__(self, model="text-davinci-003", temperature=0.7, max_messages=6):
        self.llm = OpenAI(
            temperature=temperature,
            model_name=model,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        # ConversationBufferMemory stores the complete message history in memory.
        # memory_key must match the history variable expected by ConversationChain.
        # return_messages=False keeps the history as a plain string, which is
        # compatible with the default ConversationChain prompt and OpenAI models.
        self.memory = ConversationBufferMemory(
            memory_key="history",
            input_key="input",
            return_messages=False,
        )
        self.max_messages = max_messages
        self.chain = ConversationChain(llm=self.llm, memory=self.memory, verbose=True)

    def ask(self, prompt: str) -> str:
        """Send a prompt to the conversation and return the AI response."""
        self.trim_history()
        return self.chain.run(input=prompt)

    def clear_history(self) -> None:
        """Clear the conversation memory for this session."""
        self.memory.clear()

    def trim_history(self) -> None:
        """Trim the conversation memory to the last max_messages messages."""
        messages = self.memory.chat_memory.messages
        if len(messages) > self.max_messages:
            # Keep only the most recent messages.
            self.memory.chat_memory.messages = messages[-self.max_messages :]

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

    def __init__(self, model="text-davinci-003", temperature=0.7, max_summary_chars=500):
        self.llm = OpenAI(
            temperature=temperature,
            model_name=model,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.memory = ConversationSummaryMemory(llm=self.llm)
        self.max_summary_chars = max_summary_chars
        self.chain = ConversationChain(llm=self.llm, memory=self.memory, verbose=True)

    def ask(self, prompt: str) -> str:
        """Send a prompt to the conversation and return the AI response."""
        self.trim_history()
        return self.chain.run(input=prompt)

    def clear_history(self) -> None:
        """Clear the conversation memory for this session."""
        self.memory.clear()

    def trim_history(self) -> None:
        """Trim the conversation summary to a maximum number of characters."""
        if len(self.memory.summary) > self.max_summary_chars:
            self.memory.summary = self.memory.summary[: self.max_summary_chars].rstrip() + "..."

    def format_history(self) -> str:
        """Return a formatted string of the conversation summary."""
        memory_vars = self.memory.load_memory_variables({})
        history = memory_vars.get("history", "")
        if not history:
            return "No conversation history yet."
        return f"Conversation summary:\n{history}"


class RunnableConversationSession:
    """A session-scoped conversation using RunnableWithMessageHistory and an in-memory store."""

    def __init__(self, model="gpt-3.5-turbo", temperature=0.7, max_messages=6):
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
        self.max_messages = max_messages
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
        self.trim_history(session_id)
        response = self.history.invoke(
            {"input": prompt},
            config={"configurable": {"session_id": session_id}},
        )
        return response.content

    def clear_history(self, session_id: str = "default") -> None:
        """Clear the conversation memory for a session."""
        self.store.pop(session_id, None)

    def trim_history(self, session_id: str = "default") -> None:
        """Trim the conversation history for a session to the last max_messages messages."""
        history = self.store.get(session_id)
        if history and len(history.messages) > self.max_messages:
            history.messages = history.messages[-self.max_messages :]

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


def print_conversation(session, **kwargs) -> None:
    """Print the conversation history for a session.

    Args:
        session: A session object with a format_history method.
        **kwargs: Additional arguments forwarded to format_history, such as session_id.
    """
    print("\n" + session.format_history(**kwargs) + "\n")


def run_conversation(session, **kwargs) -> None:
    """Run a standard two-turn conversation and demonstrate memory clearing.

    The helper clears any existing history before starting, so each example
    starts with a clean slate. It then runs two turns, prints the stored
    history, clears it, and asks a follow-up to show that the model no longer
    remembers the earlier context. Clearing at the end also prevents stale
    state from leaking into the next example run.
    """
    session.clear_history(**kwargs)

    print("AI:", session.ask("My favorite color is blue.", **kwargs))
    print("AI:", session.ask("What is my favorite color?", **kwargs))

    print_conversation(session, **kwargs)

    session.clear_history(**kwargs)
    print("\nHistory cleared. The AI now remembers nothing from the previous turns.\n")

    print("AI:", session.ask("What is my favorite color?", **kwargs))

    session.clear_history(**kwargs)


def chat_with_memory(session, prompts, **kwargs) -> None:
    """Run a scripted chat with memory and print a clear transcript.

    This function sends a sequence of prompts to the session, prints each
    exchange, and after every turn prints the current memory contents. It
    demonstrates how the conversation history is accumulated and used by the
    model.

    Args:
        session: A session object with ask, clear_history, and format_history methods.
        prompts: An iterable of user prompts to send.
        **kwargs: Additional arguments forwarded to session methods (e.g., session_id).
    """
    session.clear_history(**kwargs)
    print("\n--- Chat with Memory Transcript ---\n")
    for turn, prompt in enumerate(prompts, 1):
        print(f"Turn {turn}")
        print(f"Human: {prompt}")
        response = session.ask(prompt, **kwargs)
        print(f"AI: {response}")
        print(f"\nMemory after this turn:\n{session.format_history(**kwargs)}\n")
    session.clear_history(**kwargs)


def main() -> None:
    # ConversationBufferMemory retains the full chat history across turns.
    # The ConversationChain uses the configured memory_key "history" to inject
    # the stored conversation context into the prompt on every call.
    print("\n--- ConversationBufferMemory Example ---\n")

    # Each ConversationSession has its own isolated memory.
    session = ConversationSession()
    run_conversation(session)

    # ConversationSummaryMemory example with a running summary.
    print("\n--- ConversationSummaryMemory Example ---\n")
    summary_session = ConversationSummarySession()
    run_conversation(summary_session)

    # RunnableWithMessageHistory example with a simple in-memory chat history store.
    print("\n--- RunnableWithMessageHistory Example ---\n")
    runnable_session = RunnableConversationSession()
    run_conversation(runnable_session, session_id="user-1")

    # A scripted chat that prints a clear transcript with memory contents.
    print("\n--- Chat with Memory Function Example ---\n")
    chat_session = ConversationSession()
    chat_with_memory(
        chat_session,
        [
            "My favorite color is blue.",
            "What is my favorite color?",
            "I also like pizza.",
            "What do I like?",
        ],
    )


if __name__ == "__main__":
    main()
