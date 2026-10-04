# LangChain Examples

A collection of practical, runnable examples for building applications with LangChain. Each example focuses on a specific LangChain concept and is designed to be easy to read, modify, and reuse.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Examples](#examples)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Overview

LangChain is a framework for developing applications powered by language models. This repository contains a set of examples that demonstrate how to use LangChain to build everything from simple LLM calls to complex agents and retrieval-augmented generation (RAG) systems.

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/kenwith/langchain_examples.git
   cd langchain_examples
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   ```bash
   export OPENAI_API_KEY="your_openai_api_key"
   ```

4. Run an example:
   ```bash
   python examples/01_llm.py
   ```

## Examples

For the current list of examples, see the [examples directory](examples). The examples are grouped by topic and each file is self-contained.

Run any example from the repository root with:

```bash
python examples/<filename>.py
```

## Testing

The repository includes a pytest test suite. To run the tests:

```bash
pytest
```

## Contributing

Contributions are welcome! If you'd like to add an example or improve an existing one, please open an issue or submit a pull request.

## License

MIT
