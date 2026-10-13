"""Ollama-based LangGraph example demonstrating conditional edges.

| Field       | Value                                      |
|-------------|--------------------------------------------|
| Provider    | Ollama                                     |
| Model       | llama3.1 (configurable via OLLAMA_MODEL)   |
| Framework   | LangGraph                                  |
| Topic       | Conditional edges for routing              |
"""

import os
from typing import Literal, TypedDict

from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    query: str
    category: str
    response: str


def _get_model():
    """Create an Ollama-backed chat model using provider-agnostic init."""
    return init_chat_model(
        os.getenv("OLLAMA_MODEL", "llama3.1"),
        model_provider="ollama",
    )


def classify_node(state: State) -> dict:
    """Determine the category of the query."""
    query = state["query"].lower()
    if any(word in query for word in ("weather", "temperature", "forecast")):
        category = "weather"
    elif any(word in query for word in ("hello", "hi", "hey", "greetings")):
        category = "greeting"
    else:
        category = "default"
    return {"category": category}


def handle_weather(state: State) -> dict:
    """Generate a weather response using Ollama."""
    model = _get_model()
    prompt = f"Answer the following weather question concisely:\n\n{state['query']}"
    response = model.invoke(prompt).content
    return {"response": response}


def handle_greeting(state: State) -> dict:
    """Generate a greeting response using Ollama."""
    model = _get_model()
    prompt = f"Respond to the following greeting in a friendly way:\n\n{state['query']}"
    response = model.invoke(prompt).content
    return {"response": response}


def handle_default(state: State) -> dict:
    """Generate a general response using Ollama."""
    model = _get_model()
    prompt = f"Answer the following question concisely:\n\n{state['query']}"
    response = model.invoke(prompt).content
    return {"response": response}


def route_after_classify(state: State) -> Literal["weather", "greeting", "default"]:
    """Return the next node based on the classified category."""
    return state["category"]


def build_graph():
    """Build the LangGraph state graph."""
    graph = StateGraph(State)

    graph.add_node("classify", classify_node)
    graph.add_node("weather", handle_weather)
    graph.add_node("greeting", handle_greeting)
    graph.add_node("default", handle_default)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        route_after_classify,
        {
            "weather": "weather",
            "greeting": "greeting",
            "default": "default",
        },
    )
    graph.add_edge("weather", END)
    graph.add_edge("greeting", END)
    graph.add_edge("default", END)

    return graph.compile()


def main():
    """Run a small demo of the conditional-edge graph."""
    app = build_graph()

    queries = [
        "What is the weather in Paris?",
        "Hello there!",
        "Tell me a fun fact about space.",
    ]

    for query in queries:
        result = app.invoke({"query": query})
        print(f"Query: {query}")
        print(f"Category: {result['category']}")
        print(f"Response: {result['response']}")
        print()


if __name__ == "__main__":
    main()
