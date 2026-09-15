import os

from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

# Set DATA_PATH to an environment variable or default to "data"
DATA_PATH = os.getenv("DATA_PATH", "data")


def build_vectorstore() -> FAISS:
    """Build and return a FAISS vector store from documents.

    Loads documents from the directory specified by the DATA_PATH environment
    variable, or falls back to a small built-in set of example documents if the
    directory does not exist. Splits the documents into chunks, embeds them with
    OpenAI embeddings, and returns a FAISS vector store.
    """
    # If the data directory exists, load documents from it; otherwise, use built-in documents.
    if os.path.exists(DATA_PATH):
        loader = DirectoryLoader(DATA_PATH, loader_cls=TextLoader)
        documents = loader.load()
    else:
        # Fallback built-in document set so the example runs without external files.
        builtin_texts = [
            "LangChain is a framework for developing applications powered by language models.",
            "Retrieval-augmented generation (RAG) combines retrieval of relevant documents with a language model.",
            "RAG helps reduce hallucination by grounding the model on external knowledge.",
            "Vector stores like FAISS allow efficient similarity search over document embeddings.",
            "To use this example, set OPENAI_API_KEY in your environment.",
        ]
        documents = [Document(page_content=text) for text in builtin_texts]

    # Split documents into manageable chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore


def answer_question(vectorstore: FAISS, query: str) -> str:
    """Answer a query using retrieval-augmented generation.

    Sets up a retriever from the provided vector store and a RetrievalQA chain
    using OpenAI, then runs the query and returns the answer.
    """
    # Set up the retriever and QA chain
    retriever = vectorstore.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY")),
        chain_type="stuff",
        retriever=retriever,
    )
    return qa.run(query)


# Build the vector store and answer a sample query
vectorstore = build_vectorstore()
query = "What is RAG?"
print(answer_question(vectorstore, query))
