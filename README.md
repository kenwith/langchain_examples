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

Each example is a self-contained file in the [examples directory](examples). The following table lists every example file, its category, and a brief description.

| Example File | Category | Description |
| --- | --- | --- |
| 01_basic_chains.py | Basics | Builds a simple LLM chain with a prompt template and output parser. |
| 02_basic_llm.py | Basics | Demonstrates a direct LLM call with a single prompt. |
| 03_basic_output_parser.py | Basics | Shows how to use output parsers to get structured responses. |
| 01_rag.py | RAG | Implements retrieval-augmented generation using a vector store. |
| 02_rag_hyde.py | RAG | Uses HyDE to improve retrieval quality. |
| 01_agents.py | Agents | Creates a ReAct agent that can use tools. |
| 01_langgraph.py | LangGraph | Builds a stateful workflow with conditional edges. |
| 01_async_basic.py | Async | Runs multiple LLM calls concurrently using asyncio. |
| 02_async_chain.py | Async | Composes an async chain with runnables. |
| 03_async_rag.py | Async | Performs RAG pipeline asynchronously. |
| 04_async_agents.py | Async | Runs an agent loop asynchronously. |
| 01_ollama_chat.py | Ollama | Chat with a local Ollama model. |
| 02_ollama_embeddings.py | Ollama | Generate embeddings with Ollama. |
| 03_ollama_tools.py | Ollama | Use tools with Ollama models. |
| 04_ollama_rag.py | Ollama | RAG pipeline using Ollama for generation and embeddings. |
| 05_ollama_hyde.py | Ollama | Implements HyDE with Ollama. |
| 06_ollama_rerank.py | Ollama | Reranks retrieval results with Ollama. |
| 07_ollama_structured.py | Ollama | Obtains structured output from Ollama models. |
| 08_ollama_fallback.py | Ollama | Configures fallback models with Ollama. |
| 01_async_ollama_chat.py | Async Ollama | Async chat with Ollama. |
| 02_async_ollama_embeddings.py | Async Ollama | Async embeddings with Ollama. |
| 03_async_ollama_tools.py | Async Ollama | Async tool use with Ollama. |
| 04_async_ollama_rag.py | Async Ollama | Async RAG pipeline with Ollama. |
| 05_async_ollama_hyde.py | Async Ollama | Async HyDE with Ollama. |
| 06_async_ollama_rerank.py | Async Ollama | Async reranking with Ollama. |
| 07_async_ollama_structured.py | Async Ollama | Async structured output with Ollama. |
| 08_async_ollama_fallback.py | Async Ollama | Async fallback with Ollama. |

The examples fall into the following categories:

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
