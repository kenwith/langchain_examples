"""
Example 46: Ollama Memory
==========================

| Parameter    | Value                                                                 |
|--------------|-----------------------------------------------------------------------|
| Example      | 46                                                                    |
| Title        | Ollama Memory                                                         |
| Description  | Demonstrates conversational memory using an Ollama chat model via     |
|              | `init_chat_model`.                                                    |
| Dependencies | langchain, langchain-ollama, ollama                                    |
| Environment  | `OLLAMA_MODEL` (optional), `OLLAMA_BASE_URL` (optional)               |

This example shows how to use `init_chat_model` to create an Ollama-backed
chat model and combine it with conversation memory to maintain context
across turns. The model is created provider-agnostically, so the same code
can be adapted to other providers supported by LangChain.

Run the example:

    python examples/46_ollama_memory.py

You will be prompted to enter messages. Type `exit` or `quit` to end.
"""

import os

from langchain.chains import ConversationChain
from langchain.chat_models import init_chat_model
from langchain.memory import ConversationBufferMemory


def create_chat_model():
    """Create an Ollama chat model using init_chat_model."""
    return init_chat_model(
        model=os.getenv("OLLAMA_MODEL", "llama3.1"),
        model_provider="ollama",
        base_url=os.getenv("OLLAMA_BASE_URL"),  # Defaults to http://localhost:11434
        temperature=0,
    )


def create_conversation_chain():
    """Create a conversation chain with buffer memory."""
    chat_model = create_chat_model()
    memory = ConversationBufferMemory()
    return ConversationChain(llm=chat_model, memory=memory)


def main():
    """Run a simple interactive conversation with memory."""
    chain = create_conversation_chain()
    print("Ollama Memory Example")
    print("=====================")
    print("You can start chatting. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            break
        response = chain.predict(input=user_input)
        print(f"AI: {response}\n")


if __name__ == "__main__":
    main()
