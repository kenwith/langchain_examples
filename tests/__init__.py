"""
Test package initialization with shared pytest fixtures.

Provides common fixtures for model configuration, temporary directories,
and test utilities used across the test suite.
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator

import pytest
from langchain.chat_models import init_chat_model


# ============================================================================
# Model Configuration Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def model_provider() -> str:
    """Return the model provider to use for tests.

    Can be overridden via LANGCHAIN_TEST_PROVIDER environment variable.
    Defaults to 'openai' for compatibility.
    """
    return os.environ.get("LANGCHAIN_TEST_PROVIDER", "openai")


@pytest.fixture(scope="session")
def model_name() -> str:
    """Return the model name to use for tests.

    Can be overridden via LANGCHAIN_TEST_MODEL environment variable.
    """
    return os.environ.get("LANGCHAIN_TEST_MODEL", "gpt-3.5-turbo")


@pytest.fixture(scope="session")
def model_kwargs() -> Dict[str, Any]:
    """Return common model kwargs for test models."""
    return {
        "temperature": 0,
        "max_tokens": 100,
    }


@pytest.fixture(scope="session")
def chat_model(model_provider: str, model_name: str, model_kwargs: Dict[str, Any]):
    """Create a provider-agnostic chat model for testing.

    Uses init_chat_model for provider-agnostic model initialization.
    Skips tests if model cannot be initialized (e.g., missing API keys).
    """
    try:
        model = init_chat_model(
            model=model_name,
            model_provider=model_provider,
            **model_kwargs,
        )
        return model
    except Exception as e:
        pytest.skip(f"Could not initialize {model_provider}/{model_name}: {e}")


# ============================================================================
# Temporary Directory Fixtures
# ============================================================================


@pytest.fixture(scope="function")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test isolation.

    Yields a Path object that is automatically cleaned up after the test.
    """
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture(scope="function")
def temp_file(temp_dir: Path) -> Generator[Path, None, None]:
    """Create a temporary file in the temp directory.

    Yields a Path to a file that can be written to during tests.
    """
    file_path = temp_dir / "test_file.txt"
    file_path.write_text("")
    yield file_path


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Return the path to the test data directory.

    Creates it if it doesn't exist.
    """
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


# ============================================================================
# Utility Fixtures
# ============================================================================


@pytest.fixture(scope="function")
def mock_env(monkeypatch: pytest.MonkeyPatch) -> Dict[str, str]:
    """Provide a clean environment for testing with mock variables.

    Returns a dict of the mock environment variables set.
    """
    env_vars = {
        "OPENAI_API_KEY": "test-openai-key",
        "ANTHROPIC_API_KEY": "test-anthropic-key",
        "LANGCHAIN_TEST_PROVIDER": "openai",
        "LANGCHAIN_TEST_MODEL": "gpt-3.5-turbo",
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    return env_vars


@pytest.fixture(scope="function")
def sample_text() -> str:
    """Return sample text for testing text processing functions."""
    return (
        "This is a sample text for testing purposes. "
        "It contains multiple sentences. "
        "Each sentence should be processed correctly."
    )


@pytest.fixture(scope="function")
def sample_documents() -> list:
    """Return sample documents for testing document processing."""
    from langchain_core.documents import Document

    return [
        Document(
            page_content="First document about machine learning.",
            metadata={"source": "doc1", "topic": "ml"},
        ),
        Document(
            page_content="Second document about natural language processing.",
            metadata={"source": "doc2", "topic": "nlp"},
        ),
        Document(
            page_content="Third document about computer vision.",
            metadata={"source": "doc3", "topic": "cv"},
        ),
    ]


# ============================================================================
# Pytest Configuration
# ============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (require API keys)"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_openai: marks tests that require OpenAI API key"
    )
    config.addinivalue_line(
        "markers", "requires_anthropic: marks tests that require Anthropic API key"
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list) -> None:
    """Auto-mark tests based on keywords."""
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(pytest.mark.integration)
        if "slow" in item.keywords:
            item.add_marker(pytest.mark.slow)


# ============================================================================
# Demo
# ============================================================================

if __name__ == "__main__":
    print("Test fixtures module - run with pytest to use fixtures")
    print("Available fixtures:")
    print("  - model_provider, model_name, model_kwargs, chat_model")
    print("  - temp_dir, temp_file, test_data_dir")
    print("  - mock_env, sample_text, sample_documents")
    print("\nExample usage:")
    print("  pytest tests/ -v")
    print("  LANGCHAIN_TEST_PROVIDER=anthropic pytest tests/ -v")
