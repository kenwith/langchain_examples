"""
LangChain & LangGraph Examples Package.

This package contains runnable examples that demonstrate common LangChain
and LangGraph patterns, from basic LLM calls to multi-step agent workflows.
Each example module is self-contained and focuses on a single concept or
integration, making it easy to run and adapt.

Naming conventions:
- Module names are descriptive and use lowercase words separated by underscores
  (e.g., ``example_basic_llm``).
- Shared utilities live in ``helpers`` and the CLI entry point in ``cli``.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("langchain_examples")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

# Shared helper imports for examples
from .helpers import (
    load_env,
    get_llm,
    get_embeddings,
    print_section,
    print_result,
    run_async,
)

# CLI utility
from .cli import run_example

__all__ = [
    "__version__",
    "load_env",
    "get_llm",
    "get_embeddings",
    "print_section",
    "print_result",
    "run_async",
    "run_example",
]
