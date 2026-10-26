"""
Example 79: Ollama Async LangGraph Retrieval Grader
====================================================

This example demonstrates an asynchronous LangGraph workflow that uses an
Ollama chat model to classify retrieved documents as relevant or irrelevant
to a given query. The workflow is built with plain functions and a
provider-agnostic `init_chat_model` call, allowing the model to be swapped
without changing the graph logic.

| Field       | Value                                              |
|-------------|----------------------------------------------------|
| Example     | 79 - Ollama Async LangGraph Retrieval Grader       |
| Description | Classify retrieved documents as relevant or        |
|             | irrelevant using an Ollama model in an async       |
|             | LangGraph workflow.                                |
| Model       | llama3.1 (default, configurable via env var)       |
| Provider    | ollama                                             |
| Framework   | LangGraph, LangChain                               |
| Mode        | Async                                              |

Prerequisites:
- Install LangChain, LangGraph, and Ollama.
- Start the Ollama server and pull the model.
- Set environment variables (optional):
  - `MODEL_NAME`: Ollama model name (default: `llama3.1`)
  - `MODEL_PROVIDER`: Model provider (default: `ollama`)

Usage:
    python examples/79_ollama_async_langgraph_retrieval_grader.py
"""

import asyncio
import os
from typing import List, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph


class GraphState(TypedDict):
    """State of the LangGraph workflow."""
    query: str
    documents: List[str]
    grades: List[dict]


class RelevanceGrade(TypedDict):
    """Structured output for a relevance grade."""
    score: str
    explanation: str


async def grade_document(query: str, document: str, model) -> dict:
    """Grade a single document's relevance to the query."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a grader assessing relevance of a retrieved document to a user question. "
                "If the document contains keywords or semantic meaning related to the question, "
                "grade it as relevant. Give a binary score 'yes' or 'no' to indicate whether the "
                "document is relevant to the question.",
            ),
            (
                "human",
                "Question: {query}\n\nDocument: {document}\n\n"
                "Return a JSON object with 'score' (yes/no) and 'explanation'.",
            ),
        ]
    )
    chain = prompt | model.with_structured_output(RelevanceGrade)
    result = await chain.ainvoke({"query": query, "document": document})
    return {
        "query": query,
        "document": document,
        "score": result["score"],
        "explanation": result["explanation"],
    }


async def grade_documents(state: GraphState, model) -> dict:
    """Grade all documents in the state."""
    query = state["query"]
    documents = state["documents"]
    tasks = [grade_document(query, doc, model) for doc in documents]
    grades = await asyncio.gather(*tasks)
    return {"grades": grades}


def build_graph(model):
    """Build and compile the LangGraph workflow."""
    async def grade_documents_node(state: GraphState) -> dict:
        return await grade_documents(state, model)

    workflow = StateGraph(GraphState)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.set_entry_point("grade_documents")
    workflow.add_edge("grade_documents", END)
    return workflow.compile()


async def main() -> None:
    """Run the example."""
    model = init_chat_model(
        model=os.getenv("MODEL_NAME", "llama3.1"),
        model_provider=os.getenv("MODEL_PROVIDER", "ollama"),
        temperature=0,
    )
    app = build_graph(model)

    query = "What are the benefits of using LangGraph for building agentic workflows?"
    documents = [
        "LangGraph provides a flexible framework for building stateful, multi-step agentic "
        "workflows with built-in persistence and human-in-the-loop support.",
        "The weather forecast for tomorrow is sunny with a high of 75 degrees Fahrenheit.",
        "LangGraph enables developers to define cyclic graphs, making it easier to model "
        "complex decision processes in AI agents.",
        "A recipe for chocolate chip cookies requires flour, sugar, butter, and chocolate chips.",
    ]

    result = await app.ainvoke({"query": query, "documents": documents})

    print("Query:", query)
    print("Grades:")
    for grade in result["grades"]:
        print(f"- Score: {grade['score']}")
        print(f"  Explanation: {grade['explanation']}")
        print(f"  Document: {grade['document'][:60]}...")


if __name__ == "__main__":
    asyncio.run(main())
