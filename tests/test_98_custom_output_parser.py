"""
test_98_custom_output_parser.py

Tests for the custom output parser example.

| Test | Purpose |
| --- | --- |
| test_custom_output_parser_example | Verifies that the custom output parser example executes without errors using a deterministic chat model stub. |
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType

import pytest
from langchain.chat_models import init_chat_model
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.output_parsers import BaseOutputParser


EXAMPLE_NAME = "98_custom_output_parser.py"
FAKE_RESPONSE = "title: example\ncount: 3"


def _example_path() -> Path:
    """Returns the path to the numbered custom output parser example."""
    root = Path(__file__).resolve().parents[1]
    candidates = (root / EXAMPLE_NAME, root / "examples" / EXAMPLE_NAME)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Unable to locate {EXAMPLE_NAME}")


def _load_example() -> ModuleType:
    """Loads the numbered example module without requiring a package import."""
    path = _example_path()
    spec = importlib.util.spec_from_file_location("example_98_custom_output_parser", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load example module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_model() -> FakeListChatModel:
    """Returns a deterministic chat model for offline example execution."""
    return FakeListChatModel(responses=[FAKE_RESPONSE])


def test_custom_output_parser_example(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that the custom output parser example runs without errors."""
    for target in (
        "langchain.chat_models.init_chat_model",
        "langchain.chat_models.base.init_chat_model",
    ):
        monkeypatch.setattr(target, lambda *args, **kwargs: _fake_model(), raising=False)

    example = _load_example()
    if hasattr(example, "init_chat_model"):
        monkeypatch.setattr(example, "init_chat_model", lambda *args, **kwargs: _fake_model())

    if hasattr(example, "main"):
        example.main()
        return

    parser_type = getattr(example, "CustomOutputParser", None)
    assert parser_type is not None, "The example must expose main() or CustomOutputParser"
    parsed = parser_type().parse(FAKE_RESPONSE)
    assert parsed is not None


class _DemoParser(BaseOutputParser[dict[str, str]]):
    """Minimal key-value parser used by the demonstration block."""

    def parse(self, text: str) -> dict[str, str]:
        result: dict[str, str] = {}
        for line in text.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                result[key.strip()] = value.strip()
        return result


def _demo() -> None:
    """Runs a small provider-agnostic demonstration of a custom parser."""
    model = init_chat_model(
        model=os.getenv("CHAT_MODEL", "openai:gpt-4o-mini"),
        temperature=0,
    )
    parser = _DemoParser()
    response = model.invoke("Return a title and count on separate lines.")
    print(parser.parse(response.content))


if __name__ == "__main__":
    _demo()
