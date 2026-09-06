"""
LangChain & LangGraph Examples Package
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
