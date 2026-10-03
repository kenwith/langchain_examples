import os
import numpy as np
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

def get_embedding(text, model="text-embedding-ada-002"):
    """
    Get embedding vector for the given text using OpenAI's embedding API.

    Args:
        text (str): Input text.
        model (str): Embedding model name.

    Returns:
        list: Embedding vector.
    """
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

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

    # Generate embeddings for all documents and the query
    doc_embeddings = [get_embedding(doc) for doc in documents]
    query_embedding = get_embedding(query)

    # Compute similarity scores using the reusable helper function
    similarities = []
    for i, doc_emb in enumerate(doc_embeddings):
        sim = cosine_similarity(query_embedding, doc_emb)
        similarities.append((documents[i], sim))

    # Sort by similarity score (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Print results
    print(f"Query: {query}\n")
    print("Most similar documents:")
    for doc, score in similarities:
        print(f"  ({score:.4f}) {doc}")

if __name__ == "__main__":
    main()
