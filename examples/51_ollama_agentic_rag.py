import os
import tempfile
from typing import List, Optional

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.agents.tools import Tool
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import CharacterTextSplitter

# ---------------------------------------------------------------------------
# Ollama-specific setup
# ---------------------------------------------------------------------------
# This example uses Ollama to run models locally. Before running this script:
#
#   1. Install Ollama from https://ollama.com
#   2. Pull the models used in this example:
#        ollama pull llama3.1
#        ollama pull nomic-embed-text
#   3. Make sure the Ollama service is running locally (default: http://localhost:11434)
#
# The code below uses ChatOllama for the agent LLM and OllamaEmbeddings for
# retrieval embeddings. Both connect to the local Ollama instance.
# ---------------------------------------------------------------------------

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
    """Build an agentic RAG agent using Ollama for LLM and embeddings.

    The agent uses ChatOllama with tool calling support. Tools are attached
    via ``bind_tools`` internally by ``create_tool_calling_agent``, which
    ensures the model receives the tool schemas in a consistent way.
    """
    # ChatOllama is used instead of the plain Ollama LLM so that the agent can
    # take advantage of native tool calling / bind_tools.
    llm = ChatOllama(model="llama3.1", temperature=0)

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

    # For tool-calling agents, the prompt is a chat prompt with a system
    # message, the user input, and a placeholder for the agent's scratchpad.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an assistant with access to a knowledge base. "
                "Use the KnowledgeBaseSearch tool when you need specific facts.",
            ),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    # create_tool_calling_agent binds the tools to the LLM via bind_tools,
    # making the tool schemas available to the model in a consistent format.
    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
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
