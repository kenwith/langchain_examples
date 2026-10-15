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
| 04_basic_memory.py | Basics | Adds conversation memory to a chain. |
| 05_basic_sequential.py | Basics | Chains multiple prompts together sequentially. |
| 06_basic_router.py | Basics | Routes input to different prompts based on content. |
| 07_basic_retry.py | Basics | Adds retry logic to handle transient errors. |
| 08_basic_fallback.py | Basics | Configures fallback models and chains. |
| 09_basic_cache.py | Basics | Caches LLM responses to reduce cost and latency. |
| 10_basic_structured.py | Basics | Generates structured output from a model. |
| 11_basic_tools.py | Basics | Provides tools to a model for function calling. |
| 12_basic_embeddings.py | Basics | Generates text embeddings and measures similarity. |
| 13_basic_vectorstore.py | Basics | Stores and retrieves documents with a vector store. |
| 14_rag.py | RAG | Implements retrieval-augmented generation using a vector store. |
| 15_rag_hyde.py | RAG | Uses HyDE to improve retrieval quality. |
| 16_rag_multi_query.py | RAG | Generates multiple queries for better retrieval. |
| 17_rag_rerank.py | RAG | Reranks retrieval results with a cross-encoder. |
| 18_agents.py | Agents | Creates a ReAct agent that can use tools. |
| 19_agents_tools.py | Agents | Builds an agent with custom tools and tool descriptions. |
| 20_langgraph.py | LangGraph | Builds a stateful workflow with conditional edges. |
| 21_langgraph_persistence.py | LangGraph | Persists LangGraph state across runs. |
| 22_langgraph_subgraphs.py | LangGraph | Composes workflows from subgraphs. |
| 23_langgraph_parallel.py | LangGraph | Executes LangGraph nodes in parallel. |
| 24_langgraph_human_in_the_loop.py | LangGraph | Pauses a LangGraph workflow for human approval. |
| 25_async_basic.py | Async | Runs multiple LLM calls concurrently using asyncio. |
| 26_async_chain.py | Async | Composes an async chain with runnables. |
| 27_async_rag.py | Async | Performs a RAG pipeline asynchronously. |
| 28_async_agents.py | Async | Runs an agent loop asynchronously. |
| 29_ollama_chat.py | Ollama | Chat with a local Ollama model. |
| 30_ollama_embeddings.py | Ollama | Generate embeddings with Ollama. |
| 31_ollama_tools.py | Ollama | Use tools with Ollama models. |
| 32_ollama_rag.py | Ollama | RAG pipeline using Ollama for generation and embeddings. |
| 33_ollama_hyde.py | Ollama | Implements HyDE with Ollama. |
| 34_ollama_rerank.py | Ollama | Reranks retrieval results with Ollama. |
| 35_ollama_structured.py | Ollama | Obtains structured output from Ollama models. |
| 36_ollama_fallback.py | Ollama | Configures fallback models with Ollama. |
| 37_async_ollama_chat.py | Async Ollama | Async chat with Ollama. |
| 38_async_ollama_embeddings.py | Async Ollama | Async embeddings with Ollama. |
| 39_async_ollama_tools.py | Async Ollama | Async tool use with Ollama. |
| 40_async_ollama_rag.py | Async Ollama | Async RAG pipeline with Ollama. |
| 41_async_ollama_hyde.py | Async Ollama | Async HyDE with Ollama. |
| 42_async_ollama_rerank.py | Async Ollama | Async reranking with Ollama. |
| 43_async_ollama_structured.py | Async Ollama | Async structured output with Ollama. |
| 44_async_ollama_fallback.py | Async Ollama | Async fallback with Ollama. |
| 45_async_ollama_langgraph.py | Async Ollama LangGraph | Async LangGraph workflow with Ollama. |
| 46_async_ollama_langgraph_persistence.py | Async Ollama LangGraph | Async LangGraph persistence with Ollama. |
| 47_async_ollama_langgraph_subgraphs.py | Async Ollama LangGraph | Async LangGraph subgraphs with Ollama. |
| 48_async_ollama_langgraph_parallel.py | Async Ollama LangGraph | Async parallel LangGraph nodes with Ollama. |
| 49_async_ollama_langgraph_human_in_the_loop.py | Async Ollama LangGraph | Async human-in-the-loop workflow with Ollama. |
| 50_async_ollama_langgraph_conditional.py | Async Ollama LangGraph | Async LangGraph workflow with conditional edges using Ollama. |
| 51_async_ollama_langgraph_tools.py | Async Ollama LangGraph | Async LangGraph agent with tool use and Ollama. |
| 52_async_ollama_langgraph_rag.py | Async Ollama LangGraph | Async LangGraph RAG workflow with Ollama. |
| 53_async_ollama_langgraph_retry.py | Async Ollama LangGraph | Async LangGraph workflow with retry logic using Ollama. |
| 54_async_ollama_langgraph_fallback.py | Async Ollama LangGraph | Async LangGraph workflow with model fallback using Ollama. |
| 55_async_ollama_langgraph_structured.py | Async Ollama LangGraph | Async LangGraph workflow with structured output using Ollama. |
| 56_async_ollama_langgraph_streaming.py | Async Ollama LangGraph | Async LangGraph workflow with streaming output using Ollama. |
| 57_async_ollama_langgraph_memory.py | Async Ollama LangGraph | Async LangGraph workflow with conversation memory using Ollama. |
| 58_async_ollama_langgraph_hybrid.py | Async Ollama LangGraph | Async LangGraph workflow combining local Ollama and cloud models. |
| 59_async_ollama_langgraph_multi_agent.py | Async Ollama LangGraph | Async LangGraph workflow with multiple agents using Ollama. |
| 60_async_ollama_langgraph_router.py | Async Ollama LangGraph | Async LangGraph workflow with routing logic using Ollama. |
| 61_async_ollama_langgraph_guardrails.py | Async Ollama LangGraph | Async LangGraph workflow with guardrails using Ollama. |
| 62_async_ollama_langgraph_cache.py | Async Ollama LangGraph | Async LangGraph workflow with response caching using Ollama. |
| 63_async_ollama_langgraph_embeddings.py | Async Ollama LangGraph | Async LangGraph workflow using Ollama embeddings for retrieval. |
| 64_async_ollama_langgraph_rerank.py | Async Ollama LangGraph | Async LangGraph workflow with reranking using Ollama. |
| 65_async_ollama_langgraph_hyde.py | Async Ollama LangGraph | Async LangGraph workflow with HyDE using Ollama. |
| 66_async_ollama_langgraph_multi_query.py | Async Ollama LangGraph | Async LangGraph workflow with multi-query retrieval using Ollama. |
| 67_async_ollama_langgraph_parent_document.py | Async Ollama LangGraph | Async LangGraph workflow with parent document retrieval using Ollama. |
| 68_async_ollama_langgraph_self_query.py | Async Ollama LangGraph | Async LangGraph workflow with self-query retrieval using Ollama. |
| 69_async_ollama_langgraph_contextual_compression.py | Async Ollama LangGraph | Async LangGraph workflow with contextual compression using Ollama. |
| 70_async_ollama_langgraph_vectorstore.py | Async Ollama LangGraph | Async LangGraph workflow with vector store operations using Ollama. |
| 71_async_ollama_langgraph_agent.py | Async Ollama LangGraph | Async LangGraph agent workflow with Ollama. |
| 72_async_ollama_langgraph_react.py | Async Ollama LangGraph | Async LangGraph ReAct agent workflow with Ollama. |
| 73_async_ollama_langgraph_plan_and_execute.py | Async Ollama LangGraph | Async LangGraph plan-and-execute workflow with Ollama. |
| 74_async_ollama_langgraph_reflection.py | Async Ollama LangGraph | Async LangGraph reflection workflow with Ollama. |
| 75_async_ollama_langgraph_map_reduce.py | Async Ollama LangGraph | Async LangGraph map-reduce workflow with Ollama. |
| 76_async_ollama_langgraph_send.py | Async Ollama LangGraph | Async LangGraph workflow using the Send API for dynamic fan-out with Ollama. |

The examples fall into the following categories:

- **Basics**: LLM calls, chains, memory, output parsers, retries, fallbacks, caching, structured output, tools, embeddings, and vector stores.
- **Retrieval-Augmented Generation (RAG)**: Vector stores, embeddings, document QA, HyDE, multi-query, and reranking.
- **Agents**: ReAct agents, tool use, and conversational agents.
- **LangGraph**: Workflows, state, conditional edges, persistence, subgraphs, parallel execution, and human-in-the-loop.
- **Async**: Concurrent LLM calls, async chains, async RAG, async agents, and async Ollama examples, including structured output and fallback.
- **Ollama**: Local LLM examples using [Ollama](https://ollama.ai), including chat, embeddings, tools, RAG, HyDE, reranking, structured output, and fallback. Ollama files use the `*_ollama_*.py` naming convention. Async Ollama examples are also available, combining the `*_async_*.py` and `*_ollama_*.py` naming conventions. These examples require a locally running Ollama server.
- **Async Ollama LangGraph**: Async LangGraph workflows that run entirely with Ollama models. These examples combine the `*_async_*.py`, `*_ollama_*.py`, and `*_langgraph_*.py` naming conventions and require a local Ollama server.

## Async Examples

Async examples demonstrate how to use LangChain's asynchronous API to build concurrent applications. These examples are particularly useful for workloads that involve multiple LLM calls, network requests, or streaming operations. The async examples follow the naming convention `*_async_*.py` and are located in the [examples directory](examples). They cover:

- **Concurrent LLM calls**: Running multiple model requests in parallel using `asyncio.gather`.
- **Async chains**: Using async-compatible chains and runnables to compose workflows without blocking the event loop.
- **Async RAG**: Performing retrieval and generation steps asynchronously to reduce latency in document QA pipelines.
- **Async agents**: Running agent loops and tool calls asynchronously, enabling more efficient handling of multi-step tasks.
- **Async Ollama**: Running Ollama models asynchronously for chat, embeddings, tools, RAG, HyDE, reranking, structured output, and fallback. These files follow the `*_async_ollama_*.py` naming convention and require a local Ollama server.
- **Async Ollama LangGraph**: Building stateful LangGraph workflows asynchronously with Ollama. These files follow the `*_async_ollama_langgraph_*.py` naming convention and include persistence, subgraphs, parallel execution, conditional edges, tools, RAG, retry, fallback, structured output, streaming, memory, routing, guardrails, caching, map-reduce, dynamic fan-out, and other advanced patterns.

An async example can be run with a command like:

```bash
python examples/25_async_basic.py
```

The filename should be replaced with the actual async example to run. Async Ollama examples can be run in the same way, for example:

```bash
python examples/37_async_ollama_chat.py
```

An async Ollama LangGraph example can be run similarly:

```bash
python examples/45_async_ollama_langgraph.py
```
