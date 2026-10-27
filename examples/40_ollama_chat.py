"""
Example 40: Ollama Chat with provider-agnostic init_chat_model

This example demonstrates how to use `init_chat_model` with `model_provider="ollama"`
and the `llama3.2` model. It supports the optional `OLLAMA_BASE_URL` environment
variable to specify a custom Ollama server URL.

## Prerequisites

- Ollama installed and running (default: http://localhost:11434)
- Pull the required model: `ollama pull llama3.2`

## Setup

Set the environment variable `OLLAMA_BASE_URL` if you want to use a non-default server:

```bash
export OLLAMA_BASE_URL="http://your-ollama-server:11434"
```

## Usage

Run the interactive chat loop:

```bash
python examples/40_ollama_chat.py
```

### Single Turn

The `chat_once` helper sends one prompt to the model and returns the response text:

```python
response = chat_once("What is the capital of France?")
print(response)
```

This is useful for scripting or testing a single prompt without starting an interactive session.

## Example Output

(Example output depends on the model response, e.g., "The capital of France is Paris.")

## Functions

- `main()` - the main interactive chat loop
- `chat_once(prompt)` - send a single prompt and return the response text
"""

import os

from langchain.chat_models import init_chat_model


def _create_ollama_llm():
    """Create an Ollama chat model using init_chat_model.

    Returns:
        The initialized chat model.

    Raises:
        RuntimeError: If the model cannot be initialized, with a helpful message.
    """
    model = "llama3.2"
    base_url = os.getenv("OLLAMA_BASE_URL")

    kwargs = {}
    if base_url:
        kwargs["base_url"] = base_url

    try:
        return init_chat_model(model, model_provider="ollama", **kwargs)
    except ConnectionError:
        message = "Error: Cannot connect to the Ollama server.\n"
        if base_url:
            message += f"  Tried to reach: {base_url}\n"
        else:
            message += "  Default URL: http://localhost:11434\n"
        message += "Make sure Ollama is installed and running.\n"
        message += "If it's not running, start it with: `ollama serve`"
        raise RuntimeError(message) from None
    except Exception as e:
        message = (
            f"An error occurred while initializing the model: {e}\n"
            f"Make sure the model '{model}' is pulled. You can pull it with: "
            f"`ollama pull {model}`\n"
            "Also verify that the Ollama server is reachable."
        )
        raise RuntimeError(message) from None


def chat_once(prompt: str) -> str:
    """Send a single prompt to the Ollama model and return the response text.

    Args:
        prompt: The user message to send.

    Returns:
        The model's response content as a string.

    Raises:
        RuntimeError: If the model cannot be initialized.
    """
    llm = _create_ollama_llm()
    response = llm.invoke(prompt)
    return response.content


def main():
    """Run an interactive chat loop with Ollama."""
    try:
        llm = _create_ollama_llm()
    except RuntimeError as e:
        print(e)
        return

    print("Ollama chat initialized. Type 'exit' or 'quit' to end the conversation.")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user_input.lower() in {"exit", "quit"}:
            print("Exiting.")
            break

        if not user_input:
            continue

        try:
            response = llm.invoke(user_input)
            print(f"AI: {response.content}")
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
