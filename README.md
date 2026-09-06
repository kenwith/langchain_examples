# LangChain & LangGraph Examples

A collection of provider-agnostic examples demonstrating LangChain and LangGraph patterns.

## Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template and add your API keys
cp .env.example .env
# Edit .env with your keys
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
| Ollama connection refused | Start Ollama server: `ollama serve` and pull model: `ollama pull llama3.1`; verify `http://localhost:11434` is accessible |
| Ollama model not found | Pull the model first: `ollama pull <model-name>` (e.g., `ollama pull llama3.1`, `ollama pull mistral`) |
| LangGraph state errors | Check state schema matches across nodes; use `StateGraph` type hints; ensure all nodes return dict with same keys |
| `RateLimitError` / `429` | Implement exponential backoff; check provider quota limits; consider `max_retries` in model config |
| `ContextWindowExceeded` / token limit | Reduce input length; use smaller model; implement summarization or sliding window for conversation history |
| `ImportError: cannot import name 'init_chat_model'` | Upgrade LangChain: `pip install -U langchain`; `init_chat_model` requires langchain>=0.2.0 |

## Security Notes

- **Never commit `.env`** - it's in `.gitignore`
- Use `.env.example` as a template (no real keys)
- All examples use `init_chat_model` for provider-agnostic model selection
- Set model via `LANGCHAIN_MODEL` env var or `.env`
