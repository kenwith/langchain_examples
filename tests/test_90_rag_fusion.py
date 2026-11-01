"""Unit tests for reciprocal_rank_fusion using sample documents.

This module contains plain functions that verify the behavior of the
reciprocal_rank_fusion utility from langchain.retrievers.ensemble.
"""

from langchain_core.documents import Document
from langchain.retrievers.ensemble import reciprocal_rank_fusion

# Sample documents used across tests
DOC_A = Document(page_content="The quick brown fox jumps over the lazy dog", metadata={"id": 1})
DOC_B = Document(page_content="A fast brown fox leaps over a sleepy dog", metadata={"id": 2})
DOC_C = Document(page_content="The lazy dog sleeps all day", metadata={"id": 3})
DOC_D = Document(page_content="Foxes are wild animals", metadata={"id": 4})

# README-style table of tests
# | Test Name | Description |
# |-----------|-------------|
# | test_reciprocal_rank_fusion_basic | Fuses two ranked lists and verifies ordering and content. |
# | test_reciprocal_rank_fusion_empty | Returns an empty list when input is empty. |
# | test_reciprocal_rank_fusion_single_list | Returns the same list when only one ranked list is provided. |
# | test_reciprocal_rank_fusion_duplicates | Documents appearing in multiple lists are ranked higher. |

def test_reciprocal_rank_fusion_basic():
    """Fuses two ranked lists and verifies ordering and content."""
    list1 = [DOC_A, DOC_B, DOC_C]
    list2 = [DOC_B, DOC_C, DOC_D]
    fused = reciprocal_rank_fusion([list1, list2])
    # DOC_B appears in both lists, so it should be first
    assert fused[0].page_content == DOC_B.page_content
    # DOC_C appears in both lists, so it should be second
    assert fused[1].page_content == DOC_C.page_content
    # DOC_A appears only in list1, DOC_D only in list2; both should be present
    contents = {doc.page_content for doc in fused}
    assert DOC_A.page_content in contents
    assert DOC_D.page_content in contents

def test_reciprocal_rank_fusion_empty():
    """Returns an empty list when input is empty."""
    assert reciprocal_rank_fusion([]) == []

def test_reciprocal_rank_fusion_single_list():
    """Returns the same list when only one ranked list is provided."""
    single = [DOC_A, DOC_B, DOC_C]
    fused = reciprocal_rank_fusion([single])
    assert fused == single

def test_reciprocal_rank_fusion_duplicates():
    """Documents appearing in multiple lists are ranked higher."""
    list1 = [DOC_A, DOC_B]
    list2 = [DOC_B, DOC_A]
    fused = reciprocal_rank_fusion([list1, list2])
    # Both documents appear in both lists, so scores are equal; stable sort preserves original order.
    assert fused[0].page_content == DOC_A.page_content
    assert fused[1].page_content == DOC_B.page_content

if __name__ == "__main__":
    # Small demo of reciprocal_rank_fusion with sample documents.
    # Optionally uses init_chat_model to generate a response if an API key is present.
    import os

    print("Running reciprocal_rank_fusion demo...")
    list1 = [DOC_A, DOC_B, DOC_C]
    list2 = [DOC_B, DOC_C, DOC_D]
    fused = reciprocal_rank_fusion([list1, list2])
    print("Fused documents:")
    for doc in fused:
        print(f"  {doc.page_content}")

    # Optional chat model usage (provider-agnostic via init_chat_model)
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        from langchain.chat_models import init_chat_model
        model = init_chat_model("gpt-4o-mini", api_key=api_key)
        prompt = "Summarize the following documents:\n\n" + "\n".join(doc.page_content for doc in fused)
        response = model.invoke(prompt)
        print("\nChat model response:")
        print(response.content)
    else:
        print("\nNo OPENAI_API_KEY set; skipping chat model demo.")
