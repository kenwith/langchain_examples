"""Tests for the Ollama RAG-Fusion example."""

import os
import unittest
from unittest.mock import patch, MagicMock

import ollama_rag_fusion


class TestOllamaRagFusion(unittest.TestCase):
    """Test suite for the Ollama RAG-Fusion example."""

    def setUp(self):
        """Set up common mock objects."""
        self.retriever_patcher = patch.object(
            ollama_rag_fusion, "retrieve_documents", autospec=True
        )
        self.mock_retriever = self.retriever_patcher.start()
        self.generator_patcher = patch.object(
            ollama_rag_fusion, "generate_answer", autospec=True
        )
        self.mock_generator = self.generator_patcher.start()
        self.addCleanup(self.retriever_patcher.stop)
        self.addCleanup(self.generator_patcher.stop)

    def test_rag_fusion_generates_answer_with_mocked_calls(self):
        """Test that the main pipeline calls retrieval and generation once."""
        self.mock_retriever.return_value = [
            {"content": "doc1", "score": 0.9},
            {"content": "doc2", "score": 0.8},
        ]
        self.mock_generator.return_value = "A synthesized answer."

        result = ollama_rag_fusion.run_rag_fusion("What is RAG-Fusion?")

        self.assertEqual(result, "A synthesized answer.")
        self.mock_retriever.assert_called_once()
        self.mock_generator.assert_called_once()

    def test_rag_fusion_uses_multiple_queries(self):
        """Test that retrieval is called for each generated query."""
        # Patch the query generation to return a fixed list
        with patch.object(
            ollama_rag_fusion, "generate_queries", return_value=["q1", "q2", "q3"]
        ):
            self.mock_retriever.side_effect = [
                [{"content": "d1", "score": 0.9}],
                [{"content": "d2", "score": 0.7}],
                [{"content": "d3", "score": 0.5}],
            ]
            self.mock_generator.return_value = "Answer"

            ollama_rag_fusion.run_rag_fusion("Original query")

            self.assertEqual(self.mock_retriever.call_count, 3)

    def test_rag_fusion_handles_empty_retrieval(self):
        """Test that the pipeline gracefully handles no retrieved documents."""
        self.mock_retriever.return_value = []
        self.mock_generator.return_value = "Fallback answer."

        result = ollama_rag_fusion.run_rag_fusion("Query with no results")

        self.assertEqual(result, "Fallback answer.")
        self.mock_generator.assert_called_once()

    def test_rag_fusion_uses_reciprocal_rank_fusion(self):
        """Test that the fused context is built using reciprocal rank fusion."""
        # Mock retrieval results for two queries
        self.mock_retriever.side_effect = [
            [{"content": "docA", "score": 0.9}, {"content": "docB", "score": 0.8}],
            [{"content": "docB", "score": 0.7}, {"content": "docC", "score": 0.6}],
        ]
        # Capture the context passed to the generator
        captured_context = {}

        def fake_generate(prompt, context):
            captured_context["context"] = context
            return "Answer"

        self.mock_generator.side_effect = fake_generate

        ollama_rag_fusion.run_rag_fusion("Query")

        # Verify that docB appears in the fused context (present in both lists)
        context_text = " ".join(
            doc["content"] for doc in captured_context["context"]
        )
        self.assertIn("docB", context_text)


if __name__ == "__main__":
    unittest.main()
