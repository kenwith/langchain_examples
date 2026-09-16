# LangChain & LangGraph Examples

A collection of provider-agnostic examples demonstrating LangChain and LangGraph patterns.

## Setup

Follow these steps to get the examples running locally.

### Prerequisites

- Python 3.9 or later
- `git` (to clone the repository)
- pip (included with Python)
- Optional: [Ollama](https://ollama.com/) for local models

### Step 1: Clone the repository

```bash
git clone https://github.com/kenwith/langchain_examples.git
cd langchain_examples
```

### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

### Step 3: Activate the virtual environment

```bash
# On macOS/Linux
source .venv/bin/activate

# On Windows
.venv\Scripts\activate
```

### Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Configure environment variables

```bash
cp .env.example .env
```

Open `.env` in a text editor and add the API keys for the providers you plan to use. See [API Key Configuration](#api-key-configuration) below.

### Step 6: Verify setup

Run the first example:

```bash
python examples/01_basic_chains.py
```

If everything is configured correctly, you should see a generated response from your chosen model.

## Project Structure

```text
langchain_examples/
├── examples/          # Runnable example scripts
│   ├── 01_basic_chains.py
│   ├── 02_rag.py
│   ├── 03_tools_agents.py
│   ├── 04_langgraph_workflows.py
│   ├── 05_streaming.py
│   ├── 06_evaluation.py
│   ├── 07_memory.py
│   ├── 08_batch_inference.py
│   ├── 09_parallel_tool_calls.py
│   ├── 10_async.py
│   ├── 11_async_parallel_tool_calls.py
│   ├── 12_structured_output.py
│   ├── 13_rag_fusion.py
│   ├── 14_agentic_rag.py
│   ├── 15_plan_and_execute.py
│   ├── 16_timeout_retry.py
│   └── 17_caching.py
├── tests/             # Automated tests for examples
│   ├── test_01_basic_chains.py
│   ├── test_02_rag.py
│   ├── test_03_tools_agents.py
│   ├── test_04_langgraph_workflows.py
│   ├── test_05_streaming.py
│   ├── test_06_evaluation.py
│   ├── test_07_memory.py
│   ├── test_08_batch_inference.py
│   ├── test_09_parallel_tool_calls.py
│   ├── test_10_async.py
│   ├── test_11_async_parallel_tool_calls.py
│   ├── test_12_structured_output.py
│   ├── test_13_rag_fusion.py
│   ├── test_14_agentic_rag.py
│   ├── test_15_plan_and_execute.py
│   ├── test_16_timeout_retry.py
│   └── test_17_caching.py
├── .env.example       # Template for environment variables
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## Model Configuration

All examples use `init_chat_model` for provider-agnostic model selection. Configure via environment variable or `.env`:

```bash
# Environment variable (takes precedence)
export LANGCHAIN_MODEL="anthropic/claude-3-5-sonnet"
export LANGCHAIN_MODEL="openai/gpt-4o"
export LANGCHAIN_MODEL="google/gemini-1.5-pro"
export LANGCHAIN_MODEL="groq/llama-3.1-70b-versatile"
export LANGCHAIN_MODEL="ollama/llama3.1"
```

```bash
# Or in .env file
LANGCHAIN_MODEL=anthropic/claude-3-5-sonnet
# Optional: override base URL for OpenAI-compatible APIs
LANGCHAIN_API_BASE=https://api.example.com/v1
```

See [LangChain init_chat_model providers](https://python.langchain.com/docs/integrations/chat/init_chat_model/) for full provider list and configuration options.

## API Key Configuration

The examples load API keys from your environment or `.env` file. Add only the keys for the providers you intend to use.

| Provider | Environment variable | Model format | Notes |
|----------|---------------------|--------------|-------|
| Anthropic | `ANTHROPIC_API_KEY` | `anthropic/claude-3-5-sonnet` | Key starts with `sk-ant-` |
| OpenAI | `OPENAI_API_KEY` |
