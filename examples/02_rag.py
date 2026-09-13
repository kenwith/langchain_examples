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

# Set up the retriever and QA chain
retriever = vectorstore.as_retriever()
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY")),
    chain_type="stuff",
    retriever=retriever,
)

# Run a sample query
query = "What is RAG?"
print(qa.run(query))
