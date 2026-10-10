# LangChain Examples

A collection of practical, runnable examples for building applications with LangChain. Each example focuses on a specific LangChain concept and is designed to be easy to read, modify, and reuse.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Examples](#examples)
- [Async Examples](#async-examples)
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

   The examples are provider-agnostic. You only need to configure the provider you plan to use:

   - **OpenAI**: set `OPENAI_API_KEY`
   - **Anthropic**: set `ANTHROPIC_API_KEY`
   - **Google**: set `GOOGLE_API_KEY`
   - **Ollama** (local models): no API key required. Install [Ollama](https://ollama.ai), start the Ollama server,and pull a model:
     ```bash
     ollama pull llama3
     ```
     Ollama examples require a locally running Ollama server (default `http://localhost:11434`). If Ollama is running on a different host or port, set `OLLAMA_BASE_URL`.

4. Run an example:
   ```bash
   python examples/01_basic_chains.py
   ```

## Examples

Each example is a self-contained file in the [examples directory](examples). The examples fall into the following categories:

- **Basics**: LLM calls, chains, and output parsers.
- **Retrieval-Augmented Generation (RAG)**: Vector stores, embeddings, and document QA..
- **Agents**: ReAct agents, tool use,and conversational agents..
- **LangGraph**: Workflows, state, conditional edges, persistence, subgraphs,and parallel execution..
- **Async**: Concurrent LLM calls, async chains, async RAG, and async agents..
- **Ollama**: Local LLM examples using [Ollama](https://ollama.ai), including chat, embeddings, tools, RAG, HyDE,and reranking.. Ollama files use the `*_ollama_*.py` naming convention.. These examples require a locally running Ollama server..

## Async Examples

Async examples demonstrate how to use LangChain's asynchronous API to build concurrent applications. These examples are particularly useful for workloads that involve multiple LLM calls, network requests, or streaming operations. The async examples follow the naming convention `*_async_*.py` and are located in the [examples directory](examples). They cover:

- **Concurrent LLM calls**: Running multiple model requests in parallel using `asyncio.gather`.
- **Async chains**: Using async-compatible chains and runnables to compose workflows without blocking the event loop.
- **Async RAG**: Performing retrieval and generation steps asynchronously to reduce latency in document QA pipelines..
- **Async agents**: Running agent loops and tool calls asynchronously, enabling more efficient handling of multi-step tasks..

To run an async example, use a command like:

```bash
python examples/async_01_basic.py
```

Replace the filename with the actual async example you want to run..

## Testing

The repository includes a pytest test suite.. To run the tests:

```bash
pytest
```

### Running a single test

To run a single test file, pass the file path to pytest:

```bash
pytest tests/test_01_basic_chains.py
```

To run a single test case within a file, use `::` to separate the test name:

```bash
pytest tests/test_01_basic_chains.py::test_something
```

Replace `test_something` with the actual test name..

## Contributing

Contributions are welcome.. To add an example or improve an existing one, open an issue or submit a pull request..

## License

MIT
