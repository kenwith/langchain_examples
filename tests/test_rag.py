"""
RAG (Retrieval-Augmented Generation) tests for langchain_examples.

Tests cover:
- Document loading from various sources
- Text splitting strategies
- Embedding generation and vector storage
- Retrieval with different search types
- End-to-end QA pipeline with assertions

Run: python -m pytest tests/test_rag.py -v
"""

import os
import tempfile
from pathlib import Path
from typing import List

import pytest
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
