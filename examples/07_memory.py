from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
import sys


class ChatBot:
    def __init__(self):
        self.llm = OpenAI(temperature=0)
        # ConversationBufferMemory retains the full conversation history.
        # To use a sliding window of the last k messages instead, replace this
        # with ConversationBufferWindowMemory(k=5) and import it from
        # langchain.memory. Adjust k to change the number of messages retained.
        self.memory = ConversationBufferMemory()
        self.conversation = ConversationChain(llm=self.llm, memory=self.memory)

    def ask(self, prompt: str) -> str:
        return self.conversation.predict(input=prompt)

    def clear_memory(self):
        """Clear the conversation memory."""
        self.memory.clear()


def conversation_loop():
    bot = ChatBot()
    print("Chatbot ready. Type 'exit' to quit, 'clear' to reset memory.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        if user_input.lower() == "clear":
            bot.clear_memory()
            print("Memory cleared.")
            continue
        response = bot.ask(user_input)
        print(f"Bot: {response}")


def demo_conversation():
    """Run a scripted conversation to demonstrate ConversationBufferMemory."""
    bot = ChatBot()
    prompts = [
        "Hi, my name is Alice.",
        "What is my name?",
        "What did I just tell you?",
    ]
    for prompt in prompts:
        print(f"You: {prompt}")
        print(f"Bot: {bot.ask(prompt)}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo_conversation()
    else:
        conversation_loop()
