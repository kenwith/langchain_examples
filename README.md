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

# Or with specific model
LANGCHAIN_MODEL=anthropic/claude-3-5-sonnet python examples/01_basic_chains.py
```

## Security Notes

- **Never commit `.env`** - it's in `.gitignore`
- Use `.env.example` as a template (no real keys)
- All examples use `init_chat_model` for provider-agnostic model selection
- Set model via `LANGCHAIN_MODEL` env var or `.env`