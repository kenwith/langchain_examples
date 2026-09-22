"""
# Example 32: LangGraph Reflection

| | |
|---|---|
| Title | LangGraph Reflection |
| Description | A reflection workflow that generates, critiques, and revises an answer using LangGraph and `init_chat_model`. |
| Concepts | LangGraph, Reflection, Chat Models |
| Provider | Provider-agnostic via `init_chat_model` |

## Setup

Set the following environment variables to configure the model:

- `LANGCHAIN_MODEL` – the model name (e.g., `gpt-4o`, `claude-3-5-sonnet-20240620`)
- `LANGCHAIN_MODEL_PROVIDER` – the model provider (e.g., `openai`, `anthropic`)

These are read automatically by `init_chat_model()`.

## Usage

Run the script with a sample question:

```bash
python examples/32_langgraph_reflection.py
```
"""

from functools import lru_cache
from typing import TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_model():
    """Create a chat model using provider-agnostic init_chat_model."""
    return init_chat_model()


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class ReflectionState(TypedDict):
    """State for the reflection workflow."""
    question: str
    answer: str
    critique: str
    revision_count: int
    max_revisions: int


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

generate_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Answer the following question:\n{question}"),
    ]
)

critique_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a critical reviewer. If the answer is good, reply with exactly 'PASS'. "
                   "Otherwise, provide specific suggestions for improvement."),
        ("human", "Question: {question}\n\nAnswer: {answer}"),
    ]
)

revise_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Revise your answer based on the critique."),
        ("human", "Question: {question}\n\nPrevious answer: {answer}\n\nCritique: {critique}\n\n"
                  "New answer:"),
    ]
)


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def generate_answer(state: ReflectionState) -> dict:
    """Generate an initial answer."""
    chain = generate_prompt | get_model()
    response = chain.invoke({"question": state["question"]})
    return {"answer": response.content}


def critique_answer(state: ReflectionState) -> dict:
    """Critique the current answer."""
    chain = critique_prompt | get_model()
    response = chain.invoke(
        {"question": state["question"], "answer": state["answer"]}
    )
    return {"critique": response.content.strip()}


def revise_answer(state: ReflectionState) -> dict:
    """Revise the answer based on the critique."""
    chain = revise_prompt | get_model()
    response = chain.invoke(
        {
            "question": state["question"],
            "answer": state["answer"],
            "critique": state["critique"],
        }
    )
    return {
        "answer": response.content.strip(),
        "revision_count": state["revision_count"] + 1,
    }


# ---------------------------------------------------------------------------
# Conditional edge
# ---------------------------------------------------------------------------

def should_continue(state: ReflectionState) -> str:
    """Decide whether to revise again or finish."""
    if state["critique"] == "PASS":
        return "end"
    if state["revision_count"] >= state["max_revisions"]:
        return "end"
    return "revise"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph():
    """Build and compile the reflection graph."""
    graph = StateGraph(ReflectionState)

    graph.add_node("generate", generate_answer)
    graph.add_node("critique", critique_answer)
    graph.add_node("revise", revise_answer)

    graph.set_entry_point("generate")
    graph.add_edge("generate", "critique")
    graph.add_conditional_edges(
        "critique",
        should_continue,
        {
            "revise": "revise",
            "end": END,
        },
    )
    graph.add_edge("revise", "critique")

    return graph.compile()


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = build_graph()

    initial_state: ReflectionState = {
        "question": "What is the capital of France?",
        "answer": "",
        "critique": "",
        "revision_count": 0,
        "max_revisions": 3,
    }

    result = app.invoke(initial_state)

    print("\n=== Final Answer ===")
    print(result["answer"])
    print(f"\nRevisions used: {result['revision_count']}")
