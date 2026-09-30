"""Test for the ollama_memory example module."""

import unittest
from unittest.mock import patch

from ollama_memory import main


class TestOllamaMemory(unittest.TestCase):
    """Test the main function of the ollama_memory example."""

    @patch("ollama_memory.init_chat_model")
    def test_main_runs_with_mocked_model(self, mock_init_chat_model):
        """Verify main executes without errors when the model is mocked."""
        # Configure the mock model to return a fixed response
        mock_model = mock_init_chat_model.return_value
        mock_model.invoke.return_value = "Mocked response from model"

        # Call the main function (should not raise)
        main()

        # Assert that the chat model was initialized
        mock_init_chat_model.assert_called_once()

    @patch("ollama_memory.init_chat_model")
    def test_main_uses_provider_agnostic_init(self, mock_init_chat_model):
        """Ensure init_chat_model is used (provider-agnostic approach)."""
        mock_model = mock_init_chat_model.return_value
        mock_model.invoke.return_value = "Mocked response"

        main()

        # Verify init_chat_model was called with expected parameters (if any)
        # This test ensures the pattern is followed; adjust if needed.
        # For now, just check it was called.
        self.assertTrue(mock_init_chat_model.called)


if __name__ == "__main__":
    unittest.main()
