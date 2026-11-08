import os
import numpy as np
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Threshold for considering a document relevant to a query.
SIMILARITY_THRESHOLD = 0.5

def normalize_vector(vec):
    """
    Normalize a vector to unit length.

    Args:
        vec (list or np.ndarray): Input vector.

    Returns:
        np.ndarray: Unit-length vector, or the original zero vector if its norm is zero.
    """
    v = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def cosine_similarity(vec1, vec2):
    """
    Compute the cosine similarity between two vectors.

    Args:
        vec1 (list or np.ndarray): First vector.
        vec2 (list or np.ndarray): Second vector.

    Returns:
        float: Cosine similarity between vec1 and vec2.
    """
    v1 = np.asarray(vec1)
    v2 = np.asarray(vec2)
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

def is_relevant(similarity_score, threshold=SIMILARITY_THRESHOLD):
    """
    Determine whether a similarity score meets the relevance threshold.

    Args:
        similarity_score (float): Cosine similarity score.
        threshold (float): Minimum score for relevance.

    Returns:
        bool: True if the score is at least the threshold, False otherwise.
    """
    return similarity_score >= threshold

def get_embedding(text, model="text-embedding-ada-002"):
    """
    Get a normalized embedding vector for the given text using OpenAI's embedding API.

    Args:
        text (str): Input text.
        model (str): Embedding model name.

    Returns:
        list: Unit-length embedding vector.
    """
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return normalize_vector(response.data[0].embedding).tolist()

def rank_documents_by_similarity(query, documents, embedding_model="text-embedding-ada-002"):
    """
    Rank a list of documents by cosine similarity to a query.

    Args:
        query (str): The query text.
        documents (list of str): The documents to compare against.
        embedding_model (str): Embedding model name.

    Returns:
        list of tuple: Sorted list of (document, similarity_score) in descending order.
    """
    query_embedding = get_embedding(query, model=embedding_model)
    doc_embeddings = [get_embedding(doc, model=embedding_model) for doc in documents]

    similarities = []
    for doc, doc_emb in zip(documents, doc_embeddings):
        sim = cosine_similarity(query_embedding, doc_emb)
        similarities.append((doc, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities

def main():
    # Example documents
    documents = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "LangChain makes it easy to build applications with LLMs.",
        "OpenAI provides powerful embedding models.",
    ]

    # Query for which we want to find similar documents
    query = "What is LangChain?"

    # Rank documents using the reusable helper functions
    ranked_documents = rank_documents_by_similarity(query, documents)

    # Filter to only relevant documents using the explicit threshold
    relevant_documents = [(doc, score) for doc, score in ranked_documents if is_relevant(score)]

    # Print results
    print(f"Query: {query}\n")
    if relevant_documents:
        print("Most similar documents:")
        for doc, score in relevant_documents:
            print(f"  ({score:.4f}) {doc}")
    else:
        print("No documents met the relevance threshold.")

if __name__ == "__main__":
    main()
