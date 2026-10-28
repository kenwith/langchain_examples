from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.chains import ConversationChain

class ChatBot:
    def __init__(self):
        self.llm = OpenAI(temperature=0)
        self.memory = ConversationBufferMemory()
        self.conversation = ConversationChain(llm=self.llm, memory=self.memory)

    def ask(self, prompt: str) -> str:
        return self.conversation.predict(input=prompt)

def conversation_loop():
    bot = ChatBot()
    print("Chatbot ready. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response = bot.ask(user_input)
        print(f"Bot: {response}")

if __name__ == "__main__":
    conversation_loop()
