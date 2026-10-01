"""Smoke test for the contextual compression example."""

import importlib.util
import inspect
from pathlib import Path

EXAMPLE_PATH = Path(__file__).resolve().parents[1] / "examples" / "contextual_compression.py"


def _load_example():
    """Load the example module from the examples directory."""
    spec = importlib.util.spec_from_file_location("contextual_compression", EXAMPLE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_contextual_compression_uses_expected_retriever_classes():
    """Verify the example imports and references the expected retriever classes."""
    example = _load_example()
    source = inspect.getsource(example)

    assert "ContextualCompressionRetriever" in source
    assert "LLMChainExtractor" in source
    assert "init_chat_model" in source


if __name__ == "__main__":
    test_contextual_compression_uses_expected_retriever_classes()
    print("Contextual compression smoke test passed.")
