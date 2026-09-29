# LangChain Examples

A collection of practical, runnable examples for building applications with [LangChain](https://github.com/langchain-ai/langchain). Each example focuses on a specific LangChain concept and is designed to be easy to read, modify, and reuse.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Examples](#examples)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Overview

This repository contains self-contained examples that show how to use LangChain for common LLM tasks. The examples are written in Python and can be run from the command line, used as references, or adapted into your own projects.

All example scripts live in the `examples/` directory. The list below reflects the current set of files; new examples are added regularly. These examples are continuously updated to work with the latest LangChain release and best practices.

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

| File | Description |
| --- | --- |
| [examples/01_basic_chains.py](examples/01_basic_chains.py) | Creates a basic LLM chain that sends a prompt and prints the response. |
| [examples/02_chat_models.py](examples/02_chat_models.py) | Demonstrates chat models like `ChatOpenAI` with system/assistant/human message roles. |
| [examples/03_prompt_templates.py](examples/03_prompt_templates.py) | Builds reusable prompt templates with variables, partial formatting, and message-based prompts. |
| [examples/04_output_parsers.py](examples/04_output_parsers.py) | Parses LLM output into structured data using Pydantic output parsers. |
| [examples/05_memory.py](examples/05_memory.py) | Adds conversation memory so the model can remember previous turns and keep context. |
| [examples/06_chains.py](examples/06_chains.py) | Composes multiple calls or steps into a single chain using LCEL or `LLMChain`. |
| [examples/07_agents.py](examples/07_agents.py) | Uses agents to decide which tools to call based on user input. |
| [examples/08_tools.py](examples/08_tools.py) | Defines and uses custom tools so an agent can interact with external APIs or functions. |
| [examples/09_embeddings.py](examples/09_embeddings.py) | Generates text embeddings for use in search and similarity tasks. |
| [examples/10_vector_stores.py](examples/10_vector_stores.py) | Stores embeddings in a vector store and performs similarity search. |
| [examples/11_document_question_answering.py](examples/11_document_question_answering.py) | Loads, splits, embeds documents, and answers questions over their content. |
| [examples/12_summarization.py](examples/12_summarization.py) | Summarizes long documents with `map_reduce` and `refine` chains. |
| [examples/13_rag.py](examples/13_rag.py) | Combines document retrieval with generation to answer questions from a custom knowledge base. |
| [examples/14_streaming.py](examples/14_streaming.py) | Streams LLM responses token by token for a more interactive experience. |
| [examples/15_langgraph_agent.py](examples/15_langgraph_agent.py) | Builds a reactive agent using LangGraph's graph-based state machine. |
| [examples/16_langgraph_chatbot.py](examples/16_langgraph_chatbot.py) | Creates a stateful conversational chatbot with LangGraph and conditional routing. |
| [examples/17_contextual_compression.py](examples/17_contextual_compression.py) | Compresses retrieved documents down to the information relevant to a query. |
| [examples/18_query_rewriting.py](examples/18_query_rewriting.py) | Rewrites or expands a user query into multiple search variations to improve retrieval. |
| [examples/19_hybrid_search.py](examples/19_hybrid_search.py) | Combines keyword and vector search to improve retrieval quality. |
| [examples/20_multi_query_retriever.py](examples/20_multi_query_retriever.py) | Generates multiple query variations and retrieves documents for each to improve recall. |
| [examples/21_self_query_retriever.py](examples/21_self_query_retriever.py) | Infers metadata filters from a natural language query and applies them to retrieval. |
| [examples/22_ensemble_retriever.py](examples/22_ensemble_retriever.py) | Combines multiple retrievers with weighted scores to get better results. |
| [examples/23_web_research.py](examples/23_web_research.py) | Performs web searches and synthesizes answers from web content. |
| [examples/24_sql_agent.py](examples/24_sql_agent.py) | Creates an agent that can query a SQL database using natural language. |
| [examples/25_csv_agent.py](examples/25_csv_agent.py) | Answers questions over CSV data with an agent. |
| [examples/26_pandas_agent.py](examples/26_pandas_agent.py) | Uses an agent to manipulate and analyze data with pandas. |
| [examples/27_function_calling.py](examples/27_function_calling.py) | Demonstrates OpenAI function calling to extract structured data and trigger actions. |
| [examples/28_structured_output.py](examples/28_structured_output.py) | Gets structured, typed responses from LLMs using output parsers and schemas. |
| [examples/29_async.py](examples/29_async.py) | Runs LangChain operations asynchronously for better performance. |
| [examples/30_caching.py](examples/30_caching.py) | Caches LLM responses to reduce cost and latency. |
| [examples/31_callbacks.py](examples/31_callbacks.py) | Uses callbacks to monitor and interact with LangChain execution. |
| [examples/32_token_usage.py](examples/32_token_usage.py) | Tracks and counts token usage for LLM calls. |
| [examples/33_guardrails.py](examples/33_guardrails.py) | Adds validation and safety checks to LLM outputs. |
| [examples/34_evaluation.py](examples/34_evaluation.py) | Evaluates LLM chains and agents with metrics and datasets. |
| [examples/35_langsmith.py](examples/35_langsmith.py) | Traces and monitors LangChain applications with LangSmith. |
| [examples/36_agentic_rag.py](examples/36_agentic_rag.py) | Builds a RAG agent that iteratively refines queries and retrieves documents. |
| [examples/37_graph_rag.py](examples/37_graph_rag.py) | Uses knowledge graphs to improve retrieval-augmented generation. |
| [examples/38_hyde.py](examples/38_hyde.py) | Uses Hypothetical Document Embeddings (HyDE) to improve retrieval. |
| [examples/39_ollama.py](examples/39_ollama.py) | Shows how to use Ollama as a local LLM provider. |
| [examples/40_ollama_chat.py](examples/40_ollama_chat.py) | Demonstrates a chat application using Ollama with LangChain. |

## Testing

Automated tests are located in the `tests/` directory. Run them with:

```bash
pytest tests/
```

Make sure your test dependencies are installed, for example:

```bash
pip install pytest
```

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License.
