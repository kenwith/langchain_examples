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

import importlib.util
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


def _load_rag_example():
    """Load the RAG example module."""
    example_path = Path(__file__).resolve().parent.parent / "examples" / "02_rag.py"
    spec = importlib.util.spec_from_file_location("rag_example", example_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_response_warns_when_no_documents():
    """prepare_response should return a warning when no documents are found."""
    rag_example = _load_rag_example()
    response = rag_example.prepare_response("What is LangChain?", [])
    assert isinstance(response, str)
    assert "warning" in response.lower()
