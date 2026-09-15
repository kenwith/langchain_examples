import os

from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory


class ConversationSession:
    """A session-scoped conversation with memory and a clear_history helper."""

    def __init__(self, model="text-davinci-003", temperature=0.7):
        self.llm = OpenAI(
            temperature=temperature,
            model_name=model,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
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


def main() -> None:
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


if __name__ == "__main__":
    main()
