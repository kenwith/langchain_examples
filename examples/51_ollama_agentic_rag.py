import os
import tempfile
from typing import List, Optional

from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents.tools import Tool
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import CharacterTextSplitter

# Sample knowledge base content
SAMPLE_DOCS = """
Ollama allows you to run large language models locally.
It supports many models like llama3, mistral, and phi3.
Ollama also provides embedding models for retrieval augmented generation.
Agentic RAG combines reasoning agents with retrieval tools.
The agent can decide when to search for information.
"""


def create_retriever() -> FAISS:
    """Create a FAISS vector store from sample documents using Ollama embeddings."""
    # Write sample docs to a temporary file so TextLoader can read it
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(SAMPLE_DOCS)
        temp_path = f.name

    loader = TextLoader(temp_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    texts = text_splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_documents(texts, embeddings)

    # Clean up temporary file
    os.unlink(temp_path)

    return vectorstore


def build_agent() -> AgentExecutor:
    """Build an agentic RAG agent using Ollama for LLM and embeddings."""
    llm = Ollama(model="llama3", temperature=0)

    retriever = create_retriever().as_retriever(search_kwargs={"k": 3})

    def search_tool(query: str) -> str:
        """Search the knowledge base for relevant context."""
        docs = retriever.get_relevant_documents(query)
        return "\n\n".join(doc.page_content for doc in docs)

    tools = [
        Tool(
            name="KnowledgeBaseSearch",
            func=search_tool,
            description="Searches the local knowledge base for relevant information.",
        )
    ]

    prompt = PromptTemplate.from_template(
        """You are an assistant with access to a knowledge base. Use the KnowledgeBaseSearch tool when you need specific facts.

You have access to the following tools:

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

Question: {input}
Thought: {agent_scratchpad}"""
    )

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)


def main() -> None:
    """Run a sample query through the agentic RAG pipeline."""
    agent_executor = build_agent()

    question = "What models does Ollama support and how does agentic RAG work?"
    print(f"Question: {question}\n")
    response = agent_executor.invoke({"input": question})
    print(f"\nFinal Answer: {response['output']}")


if __name__ == "__main__":
    main()
