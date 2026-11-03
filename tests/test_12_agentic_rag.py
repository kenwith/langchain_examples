"""Tests for Example 12: Agentic RAG.

| Component      | Description                                        |
|----------------|----------------------------------------------------|
| Test target    | `retriever` function in `12_agentic_rag.py`        |
| Verification   | Returns a valid LangChain retriever object         |
| Dependencies   | langchain, langchain_core                           |
"""

import importlib.util
import sys
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain_core.retrievers import BaseRetriever

EXAMPLE_FILE = Path(__file__).resolve().parents[1] / "12_agentic_rag.py"
MODULE_NAME = "12_agentic_rag"


def load_example_module():
    """Load the example module from the repository root.

    | Item       | Description                          |
    |------------|--------------------------------------|
    | Returns    | The loaded example module            |
    | Raises     | FileNotFoundError if file is missing |
    """
    if not EXAMPLE_FILE.exists():
        raise FileNotFoundError(f"Example file not found: {EXAMPLE_FILE}")
    spec = importlib.util.spec_from_file_location(MODULE_NAME, EXAMPLE_FILE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def get_retriever_function():
    """Return the retriever function from the example module.

    | Item       | Description                          |
    |------------|--------------------------------------|
    | Returns    | The `retriever` callable             |
    | Raises     | AttributeError if not found          |
    """
    example = load_example_module()
    retriever_fn = getattr(example, "retriever", None)
    if retriever_fn is None:
        for name in ("create_retriever", "get_retriever", "build_retriever"):
            if hasattr(example, name):
                retriever_fn = getattr(example, name)
                break
    if retriever_fn is None:
        raise AttributeError("No retriever function found in example module.")
    return retriever_fn


def test_retriever_function_returns_valid_retriever():
    """Test that the retriever function returns a valid retriever object.

    | Test Case                    | Expected Result                          |
    |------------------------------|------------------------------------------|
    | Call retriever function      | Returns a BaseRetriever instance         |
    | Fallback check               | Has a `get_relevant_documents` method    |
    """
    retriever_fn = get_retriever_function()
    retriever = retriever_fn()
    assert retriever is not None, "Retriever function returned None."
    if BaseRetriever is not None:
        assert isinstance(retriever, BaseRetriever), (
            f"Expected BaseRetriever, got {type(retriever).__name__}"
        )
    else:
        assert callable(getattr(retriever, "get_relevant_documents", None)), (
            "Retriever object must have a get_relevant_documents method."
        )


if __name__ == "__main__":
    # | Step | Description                              |
    # |------|------------------------------------------|
    # | 1    | Load the retriever function              |
    # | 2    | Call it and print the retriever type     |
    # | 3    | Demonstrate provider-agnostic model init |
    retriever_fn = get_retriever_function()
    retriever = retriever_fn()
    print(f"Retriever type: {type(retriever).__name__}")

    model = init_chat_model("gpt-4o-mini", temperature=0)
    print(f"Chat model: {model.__class__.__name__}")
