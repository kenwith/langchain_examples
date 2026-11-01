# examples/89_async_adaptive_rag.py

"""Async Adaptive RAG with LangGraph and provider-agnostic chat models.

This example demonstrates an asynchronous adaptive retrieval-augmented
generation (RAG) workflow built with LangGraph and `init_chat_model`.

| Component         | Description                                                     |
|-------------------|-----------------------------------------------------------------|
| Route question    | Classifies whether retrieval is needed or direct answer suffices |
| Retrieve          | Pulls candidate documents from an in-memory vector store         |
| Grade documents   | Filters retrieved documents by relevance                        |
| Rewrite           | Rephrases the question when no documents are relevant            |
| Generate          | Produces a final answer from the question and relevant context   |

The graph is provider-agnostic because the chat model is created with
`init_chat_model`, which selects a model provider from environment
variables. Embeddings for the demo vector store are deterministic and
offline so the workflow can run without external embedding services.
"""

from __future__ import annotations

import asyncio
import os
from typing import List, Literal, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

MAX_RETRIEVALS = 3
TOP_K = 4

# ---------------------------------------------------------------------------
# Model and embedding configuration
# ---------------------------------------------------------------------------

MODEL = os.getenv("MODEL", "gpt-4o-mini")
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "openai")

llm = init_chat_model(model=MODEL, model_provider=MODEL_PROVIDER, temperature=0)
embeddings = DeterministicFakeEmbeddings(size=256)

# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class GraphState(TypedDict):
    """State passed between nodes in the adaptive RAG graph."""

    question: str
    documents: List[str]
    generation: str
    loop_step: int
    datasource: str


class RouteQuery(BaseModel):
    """Structured output for routing a question."""

    datasource: Literal["retrieve", "generate"] = Field(
        description="Whether to retrieve context or generate directly"
    )


class GradeDocuments(BaseModel):
    """Structured output for grading document relevance."""

    relevant: bool = Field(
        description="Whether the document is relevant to the question"
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

ROUTE_SYSTEM_PROMPT = (
    "You are an expert at routing a user question to a vectorstore "
    "or to direct generation. If the question is about the provided "
    "domain or benefits from retrieved context, route to retrieval. "
    "Otherwise, route to direct generation."
)

GRADE_SYSTEM_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a "
    "user question. If the document contains keywords or semantic meaning "
    "related to the question, grade it as relevant."
)

REWRITE_SYSTEM_PROMPT = (
    "You are a question re-writer that converts an input question to a "
    "better version optimized for vectorstore retrieval. Return only the "
    "rewritten question with no additional text."
)

GENERATE_SYSTEM_PROMPT = (
    "You are an assistant for question-answering tasks. Use the following "
    "pieces of retrieved context to answer the question. If you do not know "
    "the answer, say that you do not know. Use three sentences maximum and "
    "keep the answer concise."
)

# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------


async def route_question(state: GraphState) -> dict:
    """Route the incoming question to retrieval or direct generation."""
    structured_llm = llm.with_structured_output(RouteQuery)
    messages = [
        SystemMessage(content=ROUTE_SYSTEM_PROMPT),
        HumanMessage(content=f"Question: {state['question']}"),
    ]
    result = await structured_llm.ainvoke(messages)
    return {"datasource": result.datasource}


async def retrieve(state: GraphState) -> dict:
    """Retrieve candidate documents from the in-memory vector store."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    docs = await retriever.ainvoke(state["question"])
    return {
        "documents": [doc.page_content for doc in docs],
        "loop_step": state["loop_step"] + 1,
    }


async def grade_documents(state: GraphState) -> dict:
    """Filter retrieved documents by relevance to the question."""
    structured_llm = llm.with_structured_output(GradeDocuments)

    async def is_relevant(document: str) -> bool:
        messages = [
            SystemMessage(content=GRADE_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Document: {document}\n\n"
                    f"Question: {state['question']}"
                )
            ),
        ]
        result = await structured_llm.ainvoke(messages)
        return result.relevant

    scores = await asyncio.gather(
        *(is_relevant(doc) for doc in state["documents"])
    )
    relevant_docs = [
        doc for doc, relevant in zip(state["documents"], scores) if relevant
    ]
    return {"documents": relevant_docs}


async def rewrite(state: GraphState) -> dict:
    """Rewrite the question for improved retrieval."""
    messages = [
        SystemMessage(content=REWRITE_SYSTEM_PROMPT),
        HumanMessage(content=f"Question: {state['question']}"),
    ]
    result = await llm.ainvoke(messages)
    return {"question": result.content.strip()}


async def generate(state: GraphState) -> dict:
    """Generate an answer from the question and any relevant context."""
    context = "\n\n".join(state["documents"]) if state["documents"] else ""
    messages = [
        SystemMessage(content=GENERATE_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Question: {state['question']}\n\n"
                f"Context:\n{context}"
            )
        ),
    ]
    result = await llm.ainvoke(messages)
    return {"generation": result.content}


# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------


def route_decision(state: GraphState) -> str:
    """Return the next node based on the routing decision."""
    return state["datasource"]


def decide_to_generate(state: GraphState) -> str:
    """Return generate when relevant documents exist or retries are spent."""
    if state["documents"]:
        return "generate"
    if state["loop_step"] < MAX_RETRIEVALS:
        return "rewrite"
    return "generate"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph():
    """Build and compile the async adaptive RAG graph."""
    graph = StateGraph(GraphState)

    graph.add_node("route_question", route_question)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("rewrite", rewrite)
    graph.add_node("generate", generate)

    graph.add_edge(START, "route_question")
    graph.add_conditional_edges(
        "route_question",
        route_decision,
        {"retrieve": "retrieve", "generate": "generate"},
    )
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {"generate": "generate", "rewrite": "rewrite"},
    )
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def _build_demo_vectorstore() -> InMemoryVectorStore:
    """Create a small in-memory vector store with sample documents."""
    documents = [
        "LangGraph is a library for building stateful, multi-agent "
        "applications with LLMs.",
        "LangChain provides standard interfaces for models, retrievers, "
        "and tools.",
        "Adaptive RAG dynamically decides whether to retrieve documents "
        "before generating an answer.",
        "Asynchronous execution in LangGraph is supported through async "
        "node functions and graph.ainvoke.",
    ]
    return InMemoryVectorStore.from_texts(documents, embedding=embeddings)


async def main() -> None:
    """Run the adaptive RAG workflow on a sample question."""
    global vectorstore
    vectorstore = _build_demo_vectorstore()

    app = build_graph()
    question = "What is LangGraph used for?"
    result = await app.ainvoke(
        {
            "question": question,
            "documents": [],
            "generation": "",
            "loop_step": 0,
            "datasource": "",
        }
    )
    print(f"Question: {question}")
    print(f"Answer: {result['generation']}")


if __name__ == "__main__":
    vectorstore = _build_demo_vectorstore()
    asyncio.run(main())
