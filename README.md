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
| 01_basic_chains.py | Basic | Builds a simple LLM chain with a prompt template and output parser. |
| 02_basic_llm.py | Basic | Demonstrates a direct LLM call with a single prompt. |
| 03_basic_output_parser.py | Basic | Shows how to use output parsers to get structured responses. |
| 04_basic_memory.py | Basic | Adds conversation memory to a chain. |
| 05_basic_sequential.py | Basic | Chains multiple prompts together sequentially. |
| 06_basic_router.py | Basic | Routes input to different prompts based on content. |
| 07_basic_retry.py | Basic | Adds retry logic to handle transient errors. |
| 08_basic_fallback.py | Basic | Configures fallback models and chains. |
| 09_basic_cache.py | Basic | Caches LLM responses to reduce cost and latency. |
| 10_basic_structured.py | Basic | Generates structured output from a model. |
| 11_basic_tools.py | Basic | Provides tools to a model for function calling. |
| 12_basic_embeddings.py | Basic | Generates text embeddings and measures similarity. |
| 13_basic_vectorstore.py | Basic | Stores and retrieves documents with a vector store. |
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
| 44_async_ollama_fallback.py | Async Ollama | Async fallback with Oll
