"""Retrieval metrics for evaluating retrieval quality.

This module implements standard information retrieval metrics including
precision@k, recall@k, MRR, hit@k, and evidence coverage.
"""

from typing import List, Set, Dict, Any


def precision_at_k(
    retrieved_ids: List[int],
    relevant_ids: Set[int],
    k: int
) -> float:
    """
    Calculate precision@k: fraction of retrieved items that are relevant.
    
    Args:
        retrieved_ids: List of retrieved chunk IDs in order
        relevant_ids: Set of relevant chunk IDs
        k: Number of top items to consider
        
    Returns:
        Precision@k score between 0.0 and 1.0
    """
    if k <= 0:
        return 0.0
    
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    
    relevant_retrieved = sum(1 for item_id in top_k if item_id in relevant_ids)
    return relevant_retrieved / len(top_k)


def recall_at_k(
    retrieved_ids: List[int],
    relevant_ids: Set[int],
    k: int
) -> float:
    """
    Calculate recall@k: fraction of relevant items that are retrieved.
    
    Args:
        retrieved_ids: List of retrieved chunk IDs in order
        relevant_ids: Set of relevant chunk IDs
        k: Number of top items to consider
        
    Returns:
        Recall@k score between 0.0 and 1.0
    """
    if not relevant_ids:
        return 1.0  # No relevant items needed, perfect recall
    
    if k <= 0:
        return 0.0
    
    top_k = retrieved_ids[:k]
    relevant_retrieved = sum(1 for item_id in top_k if item_id in relevant_ids)
    return relevant_retrieved / len(relevant_ids)


def hit_at_k(
    retrieved_ids: List[int],
    relevant_ids: Set[int],
    k: int
) -> float:
    """
    Calculate hit@k: whether at least one relevant item is in top-k.
    
    Args:
        retrieved_ids: List of retrieved chunk IDs in order
        relevant_ids: Set of relevant chunk IDs
        k: Number of top items to consider
        
    Returns:
        Hit@k score: 1.0 if at least one relevant item in top-k, else 0.0
    """
    if not relevant_ids:
        return 1.0  # No relevant items needed, perfect hit
    
    if k <= 0:
        return 0.0
    
    top_k = retrieved_ids[:k]
    return 1.0 if any(item_id in relevant_ids for item_id in top_k) else 0.0


def mean_reciprocal_rank(
    retrieved_ids: List[int],
    relevant_ids: Set[int]
) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR): reciprocal of the rank of the
    first relevant item.
    
    Args:
        retrieved_ids: List of retrieved chunk IDs in order
        relevant_ids: Set of relevant chunk IDs
        
    Returns:
        MRR score between 0.0 and 1.0
    """
    if not relevant_ids:
        return 1.0  # No relevant items needed, perfect MRR
    
    if not retrieved_ids:
        return 0.0
    
    for rank, item_id in enumerate(retrieved_ids, start=1):
        if item_id in relevant_ids:
            return 1.0 / rank
    
    return 0.0  # No relevant item found


def evidence_coverage(
    retrieved_texts: List[str],
    expected_evidence: Set[str],
    case_sensitive: bool = False
) -> float:
    """
    Calculate evidence coverage: fraction of expected evidence strings found
    in the retrieved text content.
    
    Args:
        retrieved_texts: List of retrieved text contents
        expected_evidence: Set of evidence strings that should appear
        case_sensitive: Whether matching should be case-sensitive
        
    Returns:
        Evidence coverage score between 0.0 and 1.0
    """
    if not expected_evidence:
        return 1.0  # No evidence needed, perfect coverage
    
    if not retrieved_texts:
        return 0.0
    
    # Combine all retrieved text into a single string
    combined_text = " ".join(retrieved_texts)
    if not case_sensitive:
        combined_text = combined_text.lower()
    
    # Check which evidence strings are found
    found_count = 0
    for evidence in expected_evidence:
        search_str = evidence if case_sensitive else evidence.lower()
        if search_str in combined_text:
            found_count += 1
    
    return found_count / len(expected_evidence)


def compute_all_metrics(
    retrieved_ids: List[int],
    retrieved_texts: List[str],
    relevant_ids: Set[int],
    expected_evidence: Set[str],
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, Any]:
    """
    Compute all standard retrieval metrics for a single retrieval stage.
    
    Args:
        retrieved_ids: List of retrieved chunk IDs in order
        retrieved_texts: List of retrieved text contents
        relevant_ids: Set of relevant chunk IDs
        expected_evidence: Set of evidence strings that should appear
        k_values: List of k values for precision@k, recall@k, hit@k
        
    Returns:
        Dictionary containing all computed metrics
    """
    metrics = {
        "precision_at_k": {},
        "recall_at_k": {},
        "hit_at_k": {},
        "mrr": mean_reciprocal_rank(retrieved_ids, relevant_ids),
        "evidence_coverage": evidence_coverage(retrieved_texts, expected_evidence),
    }
    
    for k in k_values:
        metrics["precision_at_k"][k] = precision_at_k(retrieved_ids, relevant_ids, k)
        metrics["recall_at_k"][k] = recall_at_k(retrieved_ids, relevant_ids, k)
        metrics["hit_at_k"][k] = hit_at_k(retrieved_ids, relevant_ids, k)
    
    # Compute missing items for diagnostics
    retrieved_set = set(retrieved_ids)
    metrics["missing_chunk_ids"] = relevant_ids - retrieved_set
    metrics["retrieved_but_unused"] = retrieved_set - relevant_ids
    
    # Compute missing evidence for diagnostics
    if expected_evidence:
        combined_text = " ".join(retrieved_texts).lower()
        metrics["missing_evidence"] = {
            ev for ev in expected_evidence if ev.lower() not in combined_text
        }
    else:
        metrics["missing_evidence"] = set()
    
    return metrics


__all__ = [
    'precision_at_k',
    'recall_at_k',
    'hit_at_k',
    'mean_reciprocal_rank',
    'evidence_coverage',
    'compute_all_metrics',
]
