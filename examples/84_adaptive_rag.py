"""Self-corrective RAG with retrieval and generation grading using LangGraph.

This example demonstrates a self-corrective RAG pipeline that uses a retrieval
grader and a generation grader to improve answer quality. The graph retrieves
documents, grades their relevance, generates an answer, and grades the answer
for groundedness. If the documents are irrelevant, the query is rewritten and
retrieval is retried. If the answer is not grounded, the query is also rewritten
and the process repeats.
"""

import logging
from typing import Literal, TypedDict

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.graph import END, START, StateGraph

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------


class GraphState(TypedDict):
    """State of the self-corrective RAG graph."""

    question: str
    documents: list[Document]
    generation: str
    iterations: int
    grounded: bool


# ---------------------------------------------------------------------------
# Structured output models for grading
# ---------------------------------------------------------------------------


class GradeDocuments(BaseModel):
    """Binary score for relevance of document to question."""

    binary_score: str = Field(description="Relevance score, 'yes' or 'no'")


class GradeGeneration(BaseModel):
    """Binary score for groundedness of answer to documents."""

    binary_score: str = Field(description="Groundedness score, 'yes' or 'no'")


# ---------------------------------------------------------------------------
# Retriever setup
# ---------------------------------------------------------------------------


def create_retriever():
    """Create a simple in-memory retriever with sample documents."""
    embeddings = init_embeddings(model="openai:text-embedding-3-small")
    docs = [
        Document(
            page_content="LangChain is a framework for developing applications "
            "powered by language models."
        ),
        Document(
            page_content="LangGraph is a library for building stateful, "
            "multi-actor applications with LLMs."
        ),
        Document(
            page_content="RAG is a technique that combines retrieval and generation."
        ),
        Document(
            page_content="Self-corrective RAG uses feedback to improve retrieval "
            "and generation."
        ),
        Document(
            page_content="The quick brown fox jumps over the lazy dog."
        ),
    ]
    vectorstore = InMemoryVectorStore.from_documents(docs, embedding=embeddings)
    return vectorstore.as_retriever()


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph(llm, retriever):
    """Build the self-corrective RAG graph."""
    max_iterations = 3

    # ------------------------------------------------------------------
    # Node: retrieve
    # ------------------------------------------------------------------
    def retrieve(state):
        """Retrieve documents based on the current question."""
        question = state["question"]
        documents = retriever.invoke(question)
        logger.info("Retrieved %d document(s) for question: %s", len(documents), question)
        return {"documents": documents}

    # ------------------------------------------------------------------
    # Node: grade_documents
    # ------------------------------------------------------------------
    def grade_documents(state):
        """Grade each document for relevance and keep only relevant ones."""
        question = state["question"]
        documents = state["documents"]
        filtered_docs = []
        grader = llm.with_structured_output(GradeDocuments)
        for doc in documents:
            prompt = (
                f"Grade the relevance of the following document to the question.\n"
                f"Question: {question}\n"
                f"Document: {doc.page_content}\n"
                f"Is it relevant? Answer 'yes' or 'no'."
            )
            score = grader.invoke(prompt)
            logger.info(
                "Document relevance: %s | document: %s",
                score.binary_score,
                doc.page_content[:80],
            )
            if score.binary_score == "yes":
                filtered_docs.append(doc)
        logger.info(
            "Retrieval grading: kept %d of %d document(s)",
            len(filtered_docs),
            len(documents),
        )
        return {"documents": filtered_docs}

    # ------------------------------------------------------------------
    # Node: transform_query
    # ------------------------------------------------------------------
    def transform_query(state):
        """Rewrite the query to improve retrieval quality."""
        question = state["question"]
        prompt = (
            f"Rewrite the following question to be more specific and likely to "
            f"retrieve relevant information.\n"
            f"Question: {question}\n"
            f"New question:"
        )
        new_question = llm.invoke(prompt).content
        logger.info("Query transform: '%s' -> '%s'", question, new_question)
        return {
            "question": new_question,
            "iterations": state["iterations"] + 1,
        }

    # ------------------------------------------------------------------
    # Node: generate
    # ------------------------------------------------------------------
    def generate(state):
        """Generate an answer from the filtered documents."""
        question = state["question"]
        documents = state["documents"]
        context = "\n\n".join([doc.page_content for doc in documents])
        prompt = (
            f"Answer the question based only on the following context.\n"
            f"Context: {context}\n"
            f"Question: {question}\n"
            f"Answer:"
        )
        logger.info("Generating answer using %d document(s)", len(documents))
        generation = llm.invoke(prompt).content
        logger.info("Generated answer: %s", generation[:120])
        return {"generation": generation}

    # ------------------------------------------------------------------
    # Node: grade_generation
    # ------------------------------------------------------------------
    def grade_generation(state):
        """Grade the answer for groundedness in the provided documents."""
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        grader = llm.with_structured_output(GradeGeneration)
        prompt = (
            f"Grade the following answer for groundedness in the provided documents.\n"
            f"Documents: {documents}\n"
            f"Answer: {generation}\n"
            f"Is the answer grounded in the documents? Answer 'yes' or 'no'."
        )
        score = grader.invoke(prompt)
        grounded = score.binary_score == "yes"
        logger.info("Groundedness grading: %s", score.binary_score)
        return {"grounded": grounded}

    # ------------------------------------------------------------------
    # Conditional edges
    # ------------------------------------------------------------------
    def decide_to_generate(state):
        """If no relevant documents, rewrite the query; otherwise generate."""
        if len(state["documents"]) == 0:
            logger.info("Routing: no relevant documents -> transform_query")
            return "transform_query"
        logger.info(
            "Routing: %d relevant document(s) -> generate",
            len(state["documents"]),
        )
        return "generate"

    def decide_to_finish(state):
        """If answer is grounded or max iterations reached, finish; else rewrite."""
        if state["grounded"] or state["iterations"] >= max_iterations:
            logger.info(
                "Routing: grounded=%s, iterations=%d -> end",
                state["grounded"],
                state["iterations"],
            )
            return "end"
        logger.info(
            "Routing: not grounded, iterations=%d -> transform_query",
            state["iterations"],
        )
        return "transform_query"

    # ------------------------------------------------------------------
    # Build graph
    # ------------------------------------------------------------------
    graph = StateGraph(GraphState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("transform_query", transform_query)
    graph.add_node("generate", generate)
    graph.add_node("grade_generation", grade_generation)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    graph.add_edge("transform_query", "retrieve")
    graph.add_edge("generate", "grade_generation")
    graph.add_conditional_edges(
        "grade_generation",
        decide_to_finish,
        {
            "end": END,
            "transform_query": "transform_query",
        },
    )

    return graph.compile()


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------


def main():
    """Run a demo query through the self-corrective RAG pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    llm = init_chat_model(model="openai:gpt-4o", temperature=0)
    retriever = create_retriever()
    app = build_graph(llm, retriever)

    question = "What is LangGraph?"
    result = app.invoke(
        {"question": question, "iterations": 0, "grounded": False}
    )
    print(result["generation"])


if __name__ == "__main__":
    main()
