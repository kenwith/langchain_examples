import asyncio
from typing import List, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import END, START, StateGraph

# Load Ollama models (make sure Ollama is running locally with these models)
llm = ChatOllama(base_url="http://localhost:11434", model="llama3.2")
embeddings = OllamaEmbeddings(base_url="http://localhost:11434", model="nomic-embed-text")

# Create an in-memory vector store with a few sample documents
vector_store = InMemoryVectorStore(embeddings)
document_texts = [
    "LangGraph is a framework for building stateful, multi-agent applications with LLMs.",
    "Adaptive RAG dynamically adjusts retrieval and generation based on document relevance.",
    "Ollama allows running open-source LLMs locally with a simple API.",
]
vector_store.add_texts(document_texts)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})


class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    retries: int


async def retrieve_node(state: GraphState) -> dict:
    """Retrieve documents relevant to the question."""
    question = state["question"]
    documents = await retriever.ainvoke(question)
    return {"documents": documents}


async def grade_documents_node(state: GraphState) -> dict:
    """Grade each retrieved document for relevance to the question."""
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []

    grading_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a meticulous document grader. Determine whether the provided document is relevant to the user's question. "
                "Respond with exactly one word: 'relevant' or 'irrelevant'.",
            ),
            ("human", "Question: {question}\n\nDocument: {document}"),
        ]
    )

    for doc in documents:
        chain = grading_prompt | llm | StrOutputParser()
        verdict = await chain.ainvoke({"question": question, "document": doc.page_content})
        if verdict.strip().lower() == "relevant":
            filtered_docs.append(doc)

    return {"documents": filtered_docs}


async def rewrite_query_node(state: GraphState) -> dict:
    """Rewrite the query to improve retrieval results after an unsuccessful retrieval."""
    question = state["question"]
    rewrite_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert query rewriter. Rewrite the user's question to improve retrieval quality. "
                "Return only the rewritten question with no additional text.",
            ),
            ("human", "Original question: {question}"),
        ]
    )
    chain = rewrite_prompt | llm | StrOutputParser()
    new_question = await chain.ainvoke({"question": question})
    return {"question": new_question.strip(), "retries": state["retries"] + 1}


async def generate_node(state: GraphState) -> dict:
    """Generate an answer from the retrieved (and graded) documents."""
    question = state["question"]
    documents = state["documents"]
    context = "\n\n".join(doc.page_content for doc in documents) if documents else "No relevant documents found."

    prompt = f"""
You are an assistant for question-answering tasks. Use the following retrieved context to answer the question.
If the context is empty or irrelevant, simply say that you don't know the answer.

Question: {question}

Context:
{context}

Answer:
""".strip()

    response = await llm.ainvoke([("system", "You are a helpful assistant."), ("human", prompt)])
    return {"generation": response.content}


def decide_to_generate(state: GraphState) -> Literal["generate", "rewrite_query"]:
    """Route to generate if there are relevant documents, otherwise rewrite the query."""
    if state.get("documents") and len(state["documents"]) > 0:
        return "generate"
    if state["retries"] < 2:
        return "rewrite_query"
    return "generate"


# Build the async LangGraph workflow
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_documents", grade_documents_node)
workflow.add_node("rewrite_query", rewrite_query_node)
workflow.add_node("generate", generate_node)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {"generate": "generate", "rewrite_query": "rewrite_query"},
)
workflow.add_edge("rewrite_query", "retrieve")
workflow.add_edge("generate", END)

app = workflow.compile()


async def main() -> None:
    question = "What is LangGraph?"
    result = await app.ainvoke({"question": question, "documents": [], "generation": "", "retries": 0})
    print("Question:", question)
    print("Answer:", result["generation"])


if __name__ == "__main__":
    asyncio.run(main())
