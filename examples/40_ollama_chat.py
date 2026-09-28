"""
Example 40: Ollama Chat with provider-agnostic init_chat_model

This example demonstrates how to use `init_chat_model` with the `ollama:llama3.2` model.
It supports the optional `OLLAMA_BASE_URL` environment variable to specify a custom Ollama server URL.

## Prerequisites

- Ollama installed and running (default: http://localhost:11434)
- Pull the model: `ollama pull llama3.2`

## Setup

Set the environment variable `OLLAMA_BASE_URL` if you want to use a non-default server:

```bash
export OLLAMA_BASE_URL="http://your-ollama-server:11434"
```

## Usage

Run the script:

```bash
python examples/40_ollama_chat.py
```

## Example Output

(Example output depends on the model response, e.g., "The capital of France is Paris.")

## Functions

- `main()` - the main demo function
"""

import os

from langchain.chat_models import init_chat_model


def main():
    """Demonstrate provider-agnostic chat with Ollama."""
    model = "ollama:llama3.2"
    base_url = os.getenv("OLLAMA_BASE_URL")

    # Pass base_url only if explicitly set, otherwise rely on default (localhost:11434)
    kwargs = {}
    if base_url:
        kwargs["base_url"] = base_url

    try:
        llm = init_chat_model(model, **kwargs)
        response = llm.invoke("What is the capital of France?")
        print(response.content)
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Make sure Ollama is running and the model 'llama3.2' is pulled.")


if __name__ == "__main__":
    main()
