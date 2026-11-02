"""
Example: RAG-Fusion with a reusable fuse_documents helper.
"""

from typing import List, Tuple, Optional
import numpy as np


def fuse_documents(
    ranked_results: List[List[Tuple[str, float]]],
    k: int = 60,
    weights: Optional[List[float]] = None,
) -> List[Tuple[str, float]]:
    """
    Fuse multiple ranked document lists using Reciprocal Rank Fusion (RRF)
    with score normalization and deduplication.

    Args:
        ranked_results: List of ranked lists, each containing (doc_id, score) tuples.
        k: Constant for RRF (default 60).
        weights: Optional weights for each ranked list.

    Returns:
        Fused and sorted list of (doc_id, fused_score).
    """
    if weights is None:
        weights = [1.0] * len(ranked_results)
    if len(weights) != len(ranked_results):
        raise ValueError("Weights must match number of ranked lists.")

    # Normalize scores in each list to [0,1] for fair combination
    normalized_lists = []
    for scores in ranked_results:
        if not scores:
            normalized_lists.append([])
            continue
        # Extract scores
        doc_ids = [doc[0] for doc in scores]
        raw_scores = np.array([doc[1] for doc in scores])
        # Normalize: min-max scaling
        min_score = raw_scores.min()
        max_score = raw_scores.max()
        if max_score == min_score:
            norm_scores = np.ones_like(raw_scores)
        else:
            norm_scores = (raw_scores - min_score) / (max_score - min_score)
        normalized_lists.append(list(zip(doc_ids, norm_scores)))

    # RRF fusion
    fused_scores = {}
    for weight, ranked_list in zip(weights, normalized_lists):
        for rank, (doc_id, score) in enumerate(ranked_list):
            # RRF contribution: weight * 1 / (k + rank + 1)
            # Also incorporate normalized score slightly for better use of raw scores.
            contribution = weight * (1.0 / (k + rank + 1) + score * 0.01)
            if doc_id in fused_scores:
                fused_scores[doc_id] += contribution
            else:
                fused_scores[doc_id] = contribution

    # Sort by fused score descending
    fused_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return fused_results


def main():
    # Example usage
    query1_results = [
        ("doc1", 0.9),
        ("doc2", 0.8),
        ("doc3", 0.7),
    ]
    query2_results = [
        ("doc2", 0.95),
        ("doc1", 0.6),
        ("doc4", 0.5),
    ]
    query3_results = [
        ("doc3", 0.85),
        ("doc4", 0.75),
        ("doc5", 0.65),
    ]

    fused = fuse_documents([query1_results, query2_results, query3_results])
    for doc_id, score in fused:
        print(f"{doc_id}: {score:.4f}")


if __name__ == "__main__":
    main()
