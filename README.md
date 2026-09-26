# LangChain Examples

A collection of practical, runnable examples for building applications with [LangChain](https://github.com/langchain-ai/langchain). Each example focuses on a specific LangChain concept and is designed to be easy to read, modify, and reuse.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Examples](#examples)
  - [01 - Basic Chains](#basic-chains)
  - [02 - Chat Models](#chat-models)
  - [03 - Prompt Templates](#prompt-templates)
  - [04 - Output Parsers](#output-parsers)
  - [05 - Memory](#memory)
  - [06 - Chains](#chains)
  - [07 - Agents](#agents)
  - [08 - Tools](#tools)
  - [09 - Embeddings](#embeddings)
  - [10 - Vector Stores](#vector-stores)
  - [11 - Document Question Answering](#document-question-answering)
  - [12 - Summarization](#summarization)
  - [13 - RAG (Retrieval-Augmented Generation)](#rag-retrieval-augmented-generation)
  - [14 - Streaming](#streaming)
  - [15 - LangGraph Agent](#langgraph-agent)
  - [16 - LangGraph Chatbot](#langgraph-chatbot)
  - [17 - Contextual Compression](#contextual-compression)
  - [18 - Query Rewriting](#query-rewriting)
  - [19 - Hybrid Search](#hybrid-search)
  - [20 - Multi-Query Retriever](#multi-query-retriever)
  - [21 - Self-Query Retriever](#self-query-retriever)
  - [22 - Ensemble Retriever](#ensemble-retriever)
  - [23 - Web Research](#web-research)
  - [24 - SQL Agent](#sql-agent)
  - [25 - CSV Agent](#csv-agent)
  - [26 - Pandas Agent](#pandas-agent)
  - [27 - Function Calling](#function-calling)
  - [28 - Structured Output](#structured-output)
  - [29 - Async](#async)
  - [30 - Caching](#caching)
  - [31 - Callbacks](#callbacks)
  - [32 - Token Usage](#token-usage)
  - [33 - Guardrails](#guardrails)
  - [34 - Evaluation](#evaluation)
  - [35 - LangSmith](#langsmith)
  - [36 - Agentic RAG](#agentic-rag)
  - [37 - Graph RAG](#graph-rag)
  - [38 - HyDE](#hyde)
- [Contributing](#contributing)
- [License](#license)

## Overview

This repository contains self-contained examples that show how to use LangChain for common LLM tasks. The examples are written in Python and can be run from the command line, used as references, or adapted into your own projects.

All example scripts live in the `examples/` directory. The list above reflects the current set of files; new examples are added regularly. These examples are continuously updated to work with the latest LangChain release and best practices.

## Quick Start

1. Clone the repository:

   ```bash
   git clone https://github.com/kenwith/langchain_examples.git
   cd langchain_examples
   ```

2. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   If there is no `requirements.txt`, install the core packages:

   ```bash
   pip install langchain openai
   ```

   For LangGraph examples, also install:

   ```bash
   pip install langgraph
   ```

4. Set your OpenAI API key:

   ```bash
   export OPENAI_API_KEY="your-api-key"   # On Windows: set OPENAI_API_KEY=your-api-key
   ```

   The examples read the key from the environment using `os.getenv("OPENAI_API_KEY")`.

5. Run an example:

   ```bash
   python examples/01_basic_chains.py
   ```

## Examples

### Basic Chains

**Script:** `examples/01_basic_chains.py`

**Run:** `python examples/01_basic_chains.py`

Shows how to create a simple LLM chain that sends a prompt to an LLM and prints the response. This is the foundation for more complex chains.

### Chat Models

**Script:** `examples/02_chat_models.py`

**Run:** `python examples/02_chat_models.py`

Demonstrates using chat models like `ChatOpenAI` for conversational interactions. Covers system/assistant/human message roles and `invoke()`.

### Prompt Templates

**Script:** `examples/03_prompt_templates.py`

**Run:** `python examples/03_prompt_templates.py`

Explains how to build reusable prompt templates with variables, partial formatting, and message-based prompt templates.

### Output Parsers

**Script:** `examples/04_output_parsers.py`

**Run:** `python examples/04_output_parsers.py`

Shows how to parse LLM output into structured data using Pydantic output parsers. Includes `StrOutputParser` and `PydanticOutputParser`.

### Memory

**Script:** `examples/05_memory.py`

**Run:** `python examples/05_memory.py`

Adds conversation memory so the model can remember previous turns and keep context. Uses `ConversationBufferMemory` and `ChatMessageHistory`.

### Chains

**Script:** `examples/06_chains.py`

**Run:** `python examples/06_chains.py`

Shows how to compose multiple calls or steps into a single LangChain chain using LCEL (LangChain Expression Language) or `LLMChain`.

### Agents

**Script:** `examples/07_agents.py`

**Run:** `python examples/07_agents.py`

Demonstrates using agents to decide which tools to call based on user input. Covers the classic `AgentExecutor` and ReAct-style agents.

### Tools

**Script:** `examples/08_tools.py`

**Run:** `python examples/08_tools.py`

Shows how to define and use custom tools so an agent can interact with external APIs or functions. Includes `@tool` decorator and `Tool` class.

### Embeddings

**Script:** `examples/09_embeddings.py`

**Run:** `python examples/09_embeddings.py`

Demonstrates generating text embeddings for use in search and similarity tasks. Uses `OpenAIEmbeddings`.

### Vector Stores

**Script:** `examples/10_vector_stores.py`

**Run:** `python examples/10_vector_stores.py`

Shows how to store embeddings in a vector store and perform similarity search. Uses `Chroma` and `FAISS`.

### Document Question Answering

**Script:** `examples/11_document_question_answering.py`

**Run:** `python examples/11_document_question_answering.py`

Shows how to load documents, split them, embed them, and answer questions over their content. Demonstrates a full document QA flow.

### Summarization

**Script:** `examples/12_summarization.py`

**Run:** `python examples/12_summarization.py`

Demonstrates summarizing long documents with LangChain. Covers the `map_reduce` and `refine` summarization chains.

### RAG (Retrieval-Augmented Generation)

**Script:** `examples/13_rag.py`

**Run:** `python examples/13_rag.py`

Combines document retrieval with generation to answer questions based on a custom knowledge base. Uses a retriever with `RetrievalQA` or LCEL.

### Streaming

**Script:** `examples/14_streaming.py`

**Run:** `python examples/14_streaming.py`

Shows how to stream responses from the LLM token by token for a more interactive experience. Uses `stream()` method.

### LangGraph Agent

**Script:** `examples/15_langgraph_agent.py`

**Run:** `python examples/15_langgraph_agent.py`

Demonstrates building a reactive agent using LangGraph's graph-based state machine. The agent can call tools and respond to user queries in a loop.

### LangGraph Chatbot

**Script:** `examples/16_langgraph_chatbot.py`

**Run:** `python examples/16_langgraph_chatbot.py`

Shows how to create a stateful conversational chatbot with LangGraph, maintaining conversation history and using conditional logic to route between nodes.

### Contextual Compression

**Script:** `examples/17_contextual_compression.py`

**Run:** `python examples/17_contextual_compression.py`

Demonstrates using a contextual compression retriever to compress retrieved documents down to the information relevant to a query, improving answer quality and reducing token usage.

### Query Rewriting

**Script:** `examples/18_query_rewriting.py`

**Run:** `python examples/18_query_rewriting.py`

Shows how to rewrite or expand a user query into multiple search variations with an LLM, improving retrieval recall and making RAG results more robust.

### Hybrid Search

**Script:** `examples/19_hybrid_search.py`

**Run:** `python examples/19_hybrid_search.py`

Demonstrates combining keyword and vector search to improve retrieval quality. Uses a `BM25Retriever` together with a dense retriever.

### Multi-Query Retriever

**Script:** `examples/20_multi_query_retriever.py`

**Run:** `python examples/20_multi_query_retriever.py`

Shows how to generate multiple query variations and retrieve documents for each to improve recall. Uses `MultiQueryRetriever`.

### Self-Query Retriever

**Script:** `examples/21_self_query_retriever.py`

**Run:** `python examples/21_self_query_retriever.py`

Demonstrates using an LLM to infer metadata filters from a natural language query and apply them to retrieval. Uses `SelfQueryRetriever`.

### Ensemble Retriever

**Script:** `examples/22_ensemble_retriever.py`

**Run:** `python examples/22_ensemble_retriever.py`

Shows how to combine multiple retrievers with weighted scores to get better results. Uses `EnsembleRetriever`.

### Web Research

**Script:** `examples/23_web_research.py`

**Run:** `python examples/23_web_research.py`

Demonstrates using LangChain to perform web searches and synthesize answers from web content. Uses `DuckDuckGoSearch` or `SERP` search tools.

### SQL Agent

**Script:** `examples/24_sql_agent.py`

**Run:** `python examples/24_sql_agent.py`

Shows how to create an agent that can query a SQL database using natural language. Uses `SQLDatabaseToolkit`.

### CSV Agent

**Script:** `examples/25_csv_agent.py`

**Run:** `python examples/25_csv_agent.py`

Demonstrates using an agent to answer questions over CSV data. Uses `CSVAgent` and `create_csv_agent`.

### Pandas Agent

**Script:** `examples/26_pandas_agent.py`

**Run:** `python examples/26_pandas_agent.py`

Shows how to use an agent to manipulate and analyze data with pandas. Uses `create_pandas_dataframe_agent`.

### Function Calling

**Script:** `examples/27_function_calling.py`

**Run:** `python examples/27_function_calling.py`

Demonstrates using OpenAI function calling to extract structured data and trigger actions. Shows how to bind functions to a chat model.

### Structured Output

**Script:** `examples/28_structured_output.py`

**Run:** `python examples/28_structured_output.py`

Shows how to get structured, typed responses from LLMs using output parsers and schemas. Uses `with_structured_output()` or a Pydantic parser.

### Async

**Script:** `examples/29_async.py`

**Run:** `python examples/29_async.py`

Demonstrates running LangChain operations asynchronously for better performance. Uses `async`/`await` and `arun()`/`ainvoke()`.

### Caching

**Script:** `examples/30_caching.py`

**Run:** `python examples/30_caching.py`

Shows how to cache LLM responses to reduce cost and latency. Uses `lgc` and `InMemoryCache`.

### Callbacks

**Script:** `examples/31_callbacks.py`

**Run:** `python examples/31_callbacks.py`

Demonstrates using callbacks to monitor and interact with LangChain execution. Uses `BaseCallbackHandler`.

### Token Usage

**Script:** `examples/32_token_usage.py`

**Run:** `python examples/32_token_usage.py`

Shows how to track and count token usage for LLM calls. Uses `TokenUsage` in callback handlers.

### Guardrails

**Script:** `examples/33_guardrails.py`

**Run:** `python examples/33_guardrails.py`

Demonstrates adding validation and safety checks to LLM outputs. Uses `OutputFixingParser` or custom validators.

### Evaluation

**Script:** `examples/34_evaluation.py`

**Run:** `python examples/34_evaluation.py`

Shows how to evaluate LLM chains and agents with metrics and datasets. Uses `EvalChain` and standard metrics.

### LangSmith

**Script:** `examples/35_langsmith.py`

**Run:** `python examples/35_langsmith.py`

Demonstrates tracing and monitoring LangChain applications with LangSmith. Sets environment variables and uses the `langsmith` SDK.

### Agentic RAG

**Script:** `examples/36_agentic_rag.py`

**Run:** `python examples/36_agentic_rag.py`

Shows how to build a retrieval-augmented generation agent that can iteratively refine queries and retrieve documents. Uses a LangGraph agent loop with a retriever tool.

### Graph RAG

**Script:** `examples/37_graph_rag.py`

**Run:** `python examples/37_graph_rag.py`

Demonstrates using knowledge graphs to improve retrieval-augmented generation. Uses a graph database (e.g., Neo4j) and text-to-Cypher.

### HyDE

**Script:** `examples/38_hyde.py`

**Run:** `python examples/38_hyde.py`

Demonstrates Hypothetical Document Embeddings (HyDE), a technique that generates a synthetic answer document from the query and uses its embedding to retrieve documents that are more likely to be relevant.

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License.
