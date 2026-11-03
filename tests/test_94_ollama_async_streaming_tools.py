"""
Test for the async streaming tools example.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch, AsyncMock

# Add the repo root to sys.path to import examples
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from langchain_core.messages import AIMessageChunk

# Import the example module
import examples.ollama_async_streaming_tools as example_module


def test_async_streaming_tools_example(capsys):
    """
    Test that the async streaming tools example runs correctly.
    """
    # Create a mock model that simulates streaming with tool calls
    mock_model = AsyncMock()
    async def fake_astream(messages, **kwargs):
        # Simulate a tool call chunk
        yield AIMessageChunk(content="", tool_calls=[{"name": "get_weather", "args": {"location": "San Francisco"}, "id": "call_1"}])
        # Simulate a final response chunk
        yield AIMessageChunk(content="The weather in San Francisco is sunny.", tool_calls=None)
    mock_model.astream = fake_astream

    # Patch the model initialization in the example to use the mock
    with patch.object(example_module, "init_chat_model", return_value=mock_model):
        # Run the example's main function
        asyncio.run(example_module.main())

    # Capture and assert output
    captured = capsys.readouterr()
    assert "The weather in San Francisco is sunny." in captured.out


if __name__ == "__main__":
    pytest.main([__file__])
