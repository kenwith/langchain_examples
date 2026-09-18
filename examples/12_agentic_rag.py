"""
Example: Agentic RAG

This example demonstrates how to combine a retriever tool with an agent
to answer questions using a ReAct loop. The agent uses a retriever tool
to fetch relevant document chunks and then reasons over them.

Prerequisites:
- Set OPENAI_API_KEY environment variable (or another supported provider)
- Install required packages:
    pip install langchain langchain-community langchain-openai faiss-cpu

Run:
    python examples/12_agentic_rag.py
"""

import os

from langchain.agents import AgentExecutor, create_react_agent
from langchain.chat_models import init_chat_model
from langchain.tools import Tool
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings

# ----------
# Sample documents
# ----------

SAMPLE_DOCUMENTS = [
    Document(
        page_content="LangChain is a framework for developing applications "
        "powered by language models.",
        metadata={"source": "langchain_framework"},
    ),
    Document(
        page_content="Agents use a language model to choose a sequence of "
        "actions to take to accomplish a goal.",
        metadata={"source": "agents_overview"},
    ),
    Document(
        page_content="Retrieval-augmented generation (RAG) combines a "
        "retrieval step with a generation step to produce answers grounded "
        "in external knowledge.",
        metadata={"source": "rag_concept"},
    ),
    Document(
        page_content="A retriever tool fetches relevant document chunks from "
        "a vector store based on a natural language query.",
        metadata={"source": "retriever_tool"},
    ),
    Document(
        page_content="The ReAct loop alternates between reasoning (Thought) "
        "and acting (Action) until a final answer is reached.",
        metadata={"source": "react_loop"},
    ),
]

# ----------
# Functions
# ----------


def create_vectorstore():
    """Create an in-memory FAISS vector store from sample documents."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.from_documents(SAMPLE_DOCUMENTS, embeddings)


def create_retriever_tool() -> Tool:
    """Create a retriever tool that searches the vector store."""
    vectorstore = create_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def search(query: str) -> str:
        docs = retriever.invoke(query)
        if not docs:
            return (
                "No relevant documents were found in the knowledge base. "
                "Politely explain that you don't have information about this "
                "topic, and suggest trying a different question or consulting "
                "the official documentation."
            )
        formatted = []
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            formatted.append(f"{doc.page_content}\n(Source: {source})")
        return "\n\n".join(formatted)

    return Tool.from_function(
        name="search_langchain_docs",
        func=search,
        description=(
            "Search the LangChain documentation for relevant context. "
            "Use this tool when you need to answer questions about LangChain, "
            "agents, RAG, or related concepts. The result includes the source "
            "name for each retrieved chunk."
        ),
    )


def build_agent() -> AgentExecutor:
    """Build a ReAct agent with a retriever tool."""
    # Use a provider-agnostic chat model. Set OPENAI_API_KEY or configure
    # another provider via init_chat_model.
    llm = init_chat_model("gpt-4o-mini", temperature=0)

    retriever_tool = create_retriever_tool()
    tools = [retriever_tool]

    prompt = PromptTemplate.from_template(
        """You are an assistant with access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

When providing the final answer, always cite the source names from the
retrieved context. For example: "According to [source name], ..." or
"Based on [source name], ...". This helps users verify the response.

Begin!

Question: {input}
Thought: {agent_scratchpad}"""
    )

    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )


def main():
    """Run a sample question through the agentic RAG system."""
    print("Building agentic RAG system...")
    agent_executor = build_agent()

    question = "What is RAG and how does it relate to agents?"
    print(f"\nQuestion: {question}\n")
    response = agent_executor.invoke({"input": question})
    print(f"\nAnswer: {response['output']}")


if __name__ == "__main__":
    main()
