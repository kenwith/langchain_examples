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

3. Set up environment variables:

   The examples are provider-agnostic. Only the provider to be used needs configuration:

   - **OpenAI**: set `OPENAI_API_KEY`
   - **Anthropic**: set `ANTHROPIC_API_KEY`
   - **Google**: set `GOOGLE_API_KEY`
   - **Ollama** (local models): no API key required. [Ollama](https://ollama.ai) must be installed, the Ollama server started, and a model pulled:
     ```bash
     ollama pull llama3
     ```
     Ollama examples require a locally running Ollama server (default `http://localhost:11434`). A different host or port can be configured by setting `OLLAMA_BASE_URL`.

4. Run an example:
   ```bash
   python examples/01_basic_chains.py
   ```

## Examples

Each example is a self-contained file in the [examples directory](examples). The examples fall into the following categories:

- **Basics**: LLM calls, chains, and output parsers.
- **Retrieval-Augmented Generation (RAG)**: Vector stores, embeddings, and document QA.
- **Agents**: ReAct agents, tool use, and conversational agents.
- **LangGraph**: Workflows, state, conditional edges, persistence, subgraphs, and parallel execution.
- **Async**: Concurrent LLM calls, async chains, async RAG, async agents, and async Ollama examples, including structured output and fallback.
- **Ollama**: Local LLM examples using [Ollama](https://ollama.ai), including chat, embeddings, tools, RAG, HyDE, reranking, structured output, and fallback. Ollama files use the `*_ollama_*.py` naming convention. Async Ollama examples are also available, combining the `*_async_*.py` and `*_ollama_*.py` naming conventions. These examples require a locally running Ollama server.

## Async Examples

Async examples demonstrate how to use LangChain's asynchronous API to build concurrent applications. These examples are particularly useful for workloads that involve multiple LLM calls, network requests, or streaming operations. The async examples follow the naming convention `*_async_*.py` and are located in the [examples directory](examples). They cover:

- **Concurrent LLM calls**: Running multiple model requests in parallel using `asyncio.gather`.
- **Async chains**: Using async-compatible chains and runnables to compose workflows without blocking the event loop.
- **Async RAG**: Performing retrieval and generation steps asynchronously to reduce latency in document QA pipelines.
- **Async agents**: Running agent loops and tool calls asynchronously, enabling more efficient handling of multi-step tasks.
- **Async Ollama**: Running Ollama models asynchronously for chat, embeddings, tools, RAG, HyDE, reranking, structured output, and fallback. These files follow the `*_async_ollama_*.py` naming convention and require a local Ollama server.

An async example can be run with a command like:

```bash
python examples/01_async_basic.py
```

The filename should be replaced with the actual async example to run. Async Ollama examples can be run in the same way, for example:

```bash
python examples/01_async_ollama_chat.py
```

## Testing

The repository includes a pytest test suite. Individual tests can be run with pytest. The full test suite can be run with:

```bash
pytest
```

A single test file can be run by passing the file path to pytest:

```bash
pytest tests/test_01_basic_chains.py
```

A single test case within a file can be run by using `::` to separate the test name:

```bash
pytest tests/test_01_basic_chains.py::test_something
```

The placeholder `test_something` should be replaced with the actual test name.

## Contributing

Contributions are welcome. Issues and pull requests can be submitted for new examples or improvements to existing examples.

## License

MIT
