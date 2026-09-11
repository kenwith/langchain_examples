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
│   └── 11_parallel_tool_calls.py
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
| OpenAI | `OPENAI_API_KEY` | `openai/gpt-4o` | Key starts with `sk-` |
| Google | `GOOGLE_API_KEY` | `google/gemini-1.5-pro` | Enable Generative Language API in Google Cloud Console |
| Groq | `GROQ_API_KEY` | `groq/llama-3.1-70b-versatile` | Key starts with `gsk_` |
| Ollama | No key required | `ollama/llama3.1` | Requires local server at `http://localhost:11434` |

Example `.env` entries:

```bash
# Choose your default model
LANGCHAIN_MODEL=anthropic/claude-3-5-sonnet

# Provider API keys (only add the ones you need)
ANTHROPIC_API_KEY=your-anthropic-api-key-here
OPENAI_API_KEY=your-openai-api-key-here
GOOGLE_API_KEY=your-google-api-key-here
GROQ_API_KEY=your-groq-api-key-here
```

Never commit real API keys. The `.env` file is listed in `.gitignore`.

## Examples

| Category | File | Description |
|----------|------|-------------|
| Basic Chains | `examples/01_basic_chains.py` | LLM + prompt + output parser |
| RAG | `examples/02_rag.py` | Retrieval-augmented generation |
| Tools/Agents | `examples/03_tools_agents.py` | Function calling, ReAct agent |
| LangGraph Workflows | `examples/04_langgraph_workflows.py` | Stateful multi-step workflows |
| Streaming | `examples/05_streaming.py` | Streaming responses |
| Evaluation | `examples/06_evaluation.py` | Testing and evaluation patterns |
| Memory | `examples/07_memory.py` | Conversation history management |
| Batch Inference | `examples/08_batch_inference.py` | Process multiple inputs efficiently with batch API calls |
| Tools/Agents | `examples/09_parallel_tool_calls.py` | Execute multiple tool calls in parallel with a single model response |
| Async | `examples/10_async.py` | Async/await patterns for concurrent model calls |
| Tools/Agents | `examples/11_parallel_tool_calls.py` | Parallel tool calls with async execution |

## Usage Examples

### 01_basic_chains.py

Demonstrates a basic LLM chain with a prompt template and output parser.

```bash
python examples/01_basic_chains.py
```

To override the model for this run:

```bash
LANGCHAIN_MODEL=openai/gpt-4o python examples/01_basic_chains.py
```

Expected output: a parsed, structured response generated by the model based on the prompt.

### 02_rag.py

Shows retrieval-augmented generation using a vector store and document loaders.

```bash
python examples/02_rag.py
```

If the script uses external documents, make sure the data files are in the expected location (check the script for `DATA_PATH` or similar). You can also modify the script to load your own documents.

### 03_tools_agents.py

Demonstrates function calling and a ReAct agent with tools.

```bash
python examples/03_tools_agents.py
```

The agent will use the defined tools to answer a question. If the script requires network access for tools like search, ensure your environment allows it.

### 04_langgraph_workflows.py

Builds a stateful multi-step workflow with LangGraph.

```bash
python examples/04_langgraph_workflows.py
```

This example shows how to define nodes, edges, and state transitions in a LangGraph workflow. Review the output to see each step execute in order.

### 05_streaming.py

Streams model responses token-by-token.

```bash
python examples/05_streaming.py
```

Some providers require explicit streaming support. If streaming does not work, check the troubleshooting section below.

### 06_evaluation.py

Runs evaluation and testing patterns for LLM outputs.

```bash
python examples/06_evaluation.py
```

This script may use criteria-based evaluation or comparison metrics. Adjust the evaluation criteria inside the script to fit your use case.

### 07_memory.py

Shows conversation history management with memory.

```bash
python examples/07_memory.py
```

The script maintains a chat history and uses it to provide context in a multi-turn conversation.

### 08_batch_inference.py

Demonstrates batch inference for processing multiple inputs efficiently.

```bash
python examples/08_batch_inference.py
```

This script shows how to send multiple prompts in a single API call using batch endpoints, reducing latency and cost.

### 09_parallel_tool_calls.py

Demonstrates parallel tool calling, where the model requests multiple tool invocations in a single response.

```bash
python examples/09_parallel_tool_calls.py
```

This example shows how to handle multiple tool calls from one model response, execute them concurrently, and feed the results back to the model.

### 10_async.py

Demonstrates async/await patterns for concurrent model calls.

```bash
python examples/10_async.py
```

This example shows how to run multiple model calls concurrently using `asyncio` and LangChain's async methods.

### 11_parallel_tool_calls.py

Demonstrates parallel tool calls with async execution.

```bash
python examples/11_parallel_tool_calls.py
```

This example shows how to combine async/await with parallel tool calls for concurrent tool execution.

## Running Examples

```bash
# Run any example
python examples/01_basic_chains.py

# Or with specific model (overrides .env)
LANGCHAIN_MODEL=anthropic/claude-3-5-sonnet python examples/01_basic_chains.py
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: langchain_anthropic` | `pip install langchain-anthropic` |
| `ModuleNotFoundError: langchain_openai` | `pip install langchain-openai` |
| `ModuleNotFoundError: langchain_google_genai` | `pip install langchain-google-genai` |
| `ModuleNotFoundError: langchain_groq` | `pip install langchain-groq` |
| `ModuleNotFoundError: langchain_ollama` | `pip install langchain-ollama` |
| `AuthenticationError` (Anthropic) | Verify `ANTHROPIC_API_KEY` in `.env`; key must start with `sk-ant-`; check [console.anthropic.com](https://console.anthropic.com) for valid key |
| `AuthenticationError` (OpenAI) | Verify `OPENAI_API_KEY` in `.env`; key must start with `sk-`; check [platform.openai.com](https://platform.openai.com) for valid key and billing |
| `AuthenticationError` (Google) | Verify `GOOGLE_API_KEY` in `.env`; enable Generative Language API in [Google Cloud Console](https://console.cloud.google.com); key must have API access |
| `AuthenticationError` (Groq) | Verify `GROQ_API_KEY` in `.env`; key must start with `gsk_`; check [console.groq.com](https://console.groq.com) for valid key |
| `init_chat_model` returns wrong model | Ensure `LANGCHAIN_MODEL` uses format `provider/model-name` (e.g., `anthropic/claude-3-5-sonnet`, `openai/gpt-4o`, `google/gemini-1.5-pro`, `groq/llama-3.1-70b-versatile`, `ollama/llama3.1`) |
| Streaming not working | Confirm model supports streaming; some providers require `streaming=True` in model kwargs; Ollama requires `streaming=True` explicitly |
| Ollama connection refused | Start Ollama server: `ollama serve` and pull model: `ollama
