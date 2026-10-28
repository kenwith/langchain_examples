"""Retrieval-augmented generation (RAG) example using LangChain.

This script builds a Chroma vector store from text files and answers questions
using a chat model with retrieval context. It can be run in two modes:

1. Create/update the vector store:
   python examples/02_rag.py --create [directory_path]

   If directory_path is omitted, the script uses built-in sample texts.

2. Query an existing vector store:
   python examples/02_rag.py

Setup:
- Install the required packages:
  pip install langchain langchain-community langchain-openai chromadb
- The OPENAI_API_KEY environment variable must be set to a valid OpenAI API key for embeddings.
- Optional: Set CHAT_MODEL and CHAT_MODEL_PROVIDER to use a different chat model. Provider-specific API keys may also need to be set.
- Optional: Place text files in a directory to use as the knowledge base.
"""
import os
import sys

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import OpenAIEmbed
