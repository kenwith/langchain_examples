"""Smoke tests for the Ollama async LangGraph reflection example.

| Test | Verifies |
|------|----------|
| test_module_imports | The example module can be imported without errors |
| test_graph_building_function_exists | The module exposes a callable graph-building function |
| test_module_syntax_is_valid | The Python source is syntactically valid |
| test_uses_provider_agnostic_chat_model_init | The module uses init_chat_model instead of hardcoded model classes |
"""

import ast
import importlib.util
from pathlib import Path

EXAMPLE_PATH = Path(__file__).resolve().parent.parent / "examples" / "ollama_async_langgraph_reflection.py"
MODULE_NAME = "ollama_async_langgraph_reflection"


def load_example_module():
    """Load the example module from its file path."""
    spec = importlib.util.spec_from_file_location(MODULE_NAME, EXAMPLE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_module_imports():
    """The example module can be imported without errors."""
    module = load_example_module()
    assert module is not None


def test_graph_building_function_exists():
    """The example module exposes a graph-building function."""
    module = load_example_module()
    graph_builder_names = ("build_graph", "create_graph")
    assert any(
        hasattr(module, name) and callable(getattr(module, name))
        for name in graph_builder_names
    ), f"Expected a graph-building function named one of {graph_builder_names}"


def test_module_syntax_is_valid():
    """The example module contains syntactically valid Python code."""
    source = EXAMPLE_PATH.read_text(encoding="utf-8")
    ast.parse(source)


def test_uses_provider_agnostic_chat_model_init():
    """The example module uses init_chat_model instead of a hardcoded model class."""
    source = EXAMPLE_PATH.read_text(encoding="utf-8")
    assert "init_chat_model" in source


if __name__ == "__main__":
    test_module_imports()
    test_graph_building_function_exists()
    test_module_syntax_is_valid()
    test_uses_provider_agnostic_chat_model_init()
    print("All smoke tests passed.")
