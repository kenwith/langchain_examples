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
│   ├── 11_parallel_tool_calls.py
│   ├── 12_structured_output.py
│   ├── 13_rag_fusion.py
│   ├── 14_agentic_rag.py
│   └── 15_plan_and_execute.py
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
│   ├── test_11_parallel_tool_calls.py
│   ├── test_12_structured_output.py
│   ├── test_13_rag_fusion.py
│   ├── test_14_agentic_rag.py
│   └── test_15_plan_and_execute.py
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
| Structured Output | `examples/12_structured_output.py` | Generate structured, typed responses with Pydantic schemas |
| Advanced RAG | `examples/13_rag_fusion.py` | Combine multiple retrieval queries for better results |
| Agentic RAG | `examples/14_agentic_rag.py` | Agent-driven retrieval-augmented generation |
| Planning | `examples/15_plan_and_execute.py` | Plan-and-Execute agent pattern with separate planning and execution phases |

## Tests

The `tests/` directory contains automated tests for each example. Tests use `pytest` and verify that the examples run correctly with mocked or minimal API calls.

### Running tests

```bash
# Run all tests
pytest tests/ -v

# Run tests for a specific example
pytest tests/test_01_basic_chains.py -v
```

Tests are designed to be run without real API keys by using fake model responses (via LangChain's `FakeListChatModel` or similar). This ensures the examples are syntactically correct and the logic works as expected in a CI environment.

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

Expected output: a parsed, structured response from the model, such as a JSON object or a string, depending on the parser used.
