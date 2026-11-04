"""Example: Ollama async reflection loop with LangGraph.

| Field       | Value                                                        |
|-------------|--------------------------------------------------------------|
| Provider    | Ollama (default) or any provider supported by init_chat_model |
| Model       | llama3.2 (default) or any model available to the provider     |
| Use case    | Critique-then-rewrite reflection loop with async graph nodes  |

This example demonstrates a reflection loop executed asynchronously with
LangGraph. The graph first generates an initial answer, then critiques that
answer, and finally rewrites it based on the critique. The critique and
rewrite steps repeat for a configurable number of iterations.

The model is created with :func:`langchain.chat_models.init_chat_model`,
which is provider-agnostic. The provider, model name, and base URL are read
from environment variables with defaults for a local Ollama instance:

- ``MODEL_PROVIDER`` (default: ``"ollama"``)
- ``MODEL_NAME`` (default: ``"llama3.2"``)
- ``MODEL_BASE_URL`` (default: ``"http://localhost:11434"``)

No API keys are required for a local Ollama server. For other providers,
set the appropriate environment variables (e.g. ``OPENAI_API_KEY``) without
hard-coding any secrets in this file.
"""

import asyncio
import os
from typing import TypedDict

from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph


class ReflectionState(TypedDict):
    """State of the reflection graph."""

    question: str
    answer: str
    critique: str
    iterations: int


def create_model():
    """Create a chat model using provider-agnostic configuration."""
    provider = os.getenv("MODEL_PROVIDER", "ollama")
    model_name = os.getenv("MODEL_NAME", "llama3.2")
    base_url = os.getenv("MODEL_BASE_URL", "http://localhost:11434")

    return init_chat_model(
        model=model_name,
        model_provider=provider,
        base_url=base_url,
    )


async def generate_initial_answer(state: ReflectionState) -> dict:
    """Generate the first answer to the question."""
    model = create_model()
    response = await model.ainvoke(
        f"Answer the following question concisely:\n\n{state['question']}"
    )
    return {
        "answer": response.content,
        "critique": "",
        "iterations": 0,
    }


async def critique_answer(state: ReflectionState) -> dict:
    """Critique the current answer, identifying weaknesses and errors."""
    model = create_model()
    prompt = (
        "You are a strict critic. Evaluate the following answer to the question.\n\n"
        f"Question: {state['question']}\n\n"
        f"Answer: {state['answer']}\n\n"
        "Provide a concise critique. Focus on factual accuracy, clarity, "
        "completeness, and style. Do not suggest specific wording; just point "
        "out what needs improvement."
    )
    response = await model.ainvoke(prompt)
    return {"critique": response.content}


async def rewrite_answer(state: ReflectionState) -> dict:
    """Rewrite the answer based on the critique."""
    model = create_model()
    prompt = (
        "Rewrite the answer to incorporate the critique and improve it.\n\n"
        f"Question: {state['question']}\n\n"
        f"Original answer: {state['answer']}\n\n"
        f"Critique: {state['critique']}\n\n"
        "Provide the revised answer only."
    )
    response = await model.ainvoke(prompt)
    return {
        "answer": response.content,
        "iterations": state["iterations"] + 1,
    }


def should_continue(state: ReflectionState) -> str:
    """Decide whether to continue the reflection loop."""
    max_iterations = int(os.getenv("MAX_REFLECTION_ITERATIONS", "2"))
    if state["iterations"] < max_iterations:
        return "critique"
    return END


async def run_reflection(question: str) -> str:
    """Run the reflection loop and return the final answer."""
    graph = StateGraph(ReflectionState)

    graph.add_node("generate", generate_initial_answer)
    graph.add_node("critique", critique_answer)
    graph.add_node("rewrite", rewrite_answer)

    graph.add_edge(START, "generate")
    graph.add_edge("generate", "critique")
    graph.add_edge("critique", "rewrite")
    graph.add_conditional_edges(
        "rewrite",
        should_continue,
        {
            "critique": "critique",
            END: END,
        },
    )

    app = graph.compile()

    final_state = await app.ainvoke(
        {
            "question": question,
            "answer": "",
            "critique": "",
            "iterations": 0,
        }
    )
    return final_state["answer"]


async def main() -> None:
    """Run the example with a sample question."""
    question = (
        "What are the main differences between classical and quantum "
        "computing? Keep the answer under 100 words."
    )
    final_answer = await run_reflection(question)
    print("Final answer:")
    print(final_answer)


if __name__ == "__main__":
    asyncio.run(main())
