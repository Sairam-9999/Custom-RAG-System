"""
Retrieval Policy Engine Module

This module provides retrieval policy building based on query analysis.
The policy determines retrieval depth, reranking usage, compression usage,
generation constraints, and answer style.
"""

import copy
from typing import Dict, Any, Optional, List


QUERY_POLICY = {
    "factual": {
        "retrieval_top_k": 4,
        "rerank_top_n": 2,
        "context_top_n": 1,
        "use_reranker": False,
        "use_context_selector": False,
        "max_new_tokens": 20,
        "use_extractive_first": True,
        "answer_style": "short_direct",
        "priority_mode": "latency",
    },
    
    "reasoning": {
        "retrieval_top_k": 12,
        "rerank_top_n": 6,
        "context_top_n": 4,
        "use_reranker": True,
        "use_context_selector": True,
        "max_new_tokens": 80,
        "use_extractive_first": False,
        "answer_style": "brief_explanation",
        "priority_mode": "balanced",
    },
    
    "comparative": {
        "retrieval_top_k": 8,
        "rerank_top_n": 8,
        "context_top_n": 4,
        "use_reranker": True,
        "use_context_selector": True,
        "max_new_tokens": 128,
        "use_extractive_first": False,
        "answer_style": "comparison",
        "priority_mode": "balanced",
    },
    
    "temporal": {
        "retrieval_top_k": 14,
        "rerank_top_n": 7,
        "context_top_n": 5,
        "use_reranker": True,
        "use_context_selector": True,
        "max_new_tokens": 100,
        "use_extractive_first": False,
        "answer_style": "timeline",
        "priority_mode": "balanced",
    },
    
    "procedural": {
        "retrieval_top_k": 10,
        "rerank_top_n": 5,
        "context_top_n": 4,
        "use_reranker": True,
        "use_context_selector": True,
        "max_new_tokens": 120,
        "use_extractive_first": False,
        "answer_style": "steps",
        "priority_mode": "balanced",
    },
    
    "analytical": {
        "retrieval_top_k": 14,
        "rerank_top_n": 7,
        "context_top_n": 5,
        "use_reranker": True,
        "use_context_selector": True,
        "max_new_tokens": 120,
        "use_extractive_first": False,
        "answer_style": "interpretation",
        "priority_mode": "quality",
    },
    
    "multi_hop": {
        "retrieval_top_k": 20,
        "rerank_top_n": 10,
        "context_top_n": 6,
        "use_reranker": True,
        "use_context_selector": True,
        "max_new_tokens": 140,
        "use_extractive_first": False,
        "answer_style": "synthesized",
        "priority_mode": "quality",
    },
    
    "unanswerable": {
        "retrieval_top_k": 6,
        "rerank_top_n": 3,
        "context_top_n": 2,
        "use_reranker": True,
        "use_context_selector": False,
        "max_new_tokens": 40,
        "use_extractive_first": False,
        "answer_style": "clarify_or_refuse",
        "priority_mode": "latency",
    },
}


def build_retrieval_policy(
    query_type: str,
    entities: Optional[List[str]] = None,
    comparison_targets: Optional[List[str]] = None,
    runtime_overrides: Optional[Dict[str, Any]] = None,
    requires_entity_coverage: bool = False,
    expected_answer_slots: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Build retrieval policy based on query type.
    
    This function converts query understanding into executable retrieval strategies.
    The policy determines:
    - retrieval depth (retrieval_top_k)
    - reranking usage (use_reranker, rerank_top_n)
    - compression usage (use_context_selector, context_top_n)
    - generation constraints (max_new_tokens, use_extractive_first)
    - answer style (answer_style)
    - latency vs quality tradeoff (priority_mode)
    
    The system is:
    - Generic: Works across domains without customization
    - Scalable: Handles complex queries with adaptive depth
    - Domain-independent: No hardcoded domain logic
    - Future-agent-compatible: Supports runtime overrides and extensions
    
    Args:
        query_type: The classified query type (factual, reasoning, etc.)
        entities: Extracted entities from the query (optional, for dynamic adjustments)
        comparison_targets: Comparison targets for comparative queries (optional)
        runtime_overrides: Optional dict to override policy parameters at runtime
        requires_entity_coverage: Whether query requires evidence for multiple answer slots
        expected_answer_slots: List of answer slots that need evidence coverage
        
    Returns:
        Dictionary of retrieval parameters with the following structure:
        {
            "retrieval_top_k": int,
            "rerank_top_n": int,
            "context_top_n": int,
            "use_reranker": bool,
            "use_context_selector": bool,
            "max_new_tokens": int,
            "use_extractive_first": bool,
            "answer_style": str,
            "priority_mode": str,
            "metadata": dict,
        }
        
    Notes:
        - Never fails on unknown query types (falls back to analytical)
        - Deep copies policies to prevent mutation
        - Supports runtime overrides for agent integration
        - Enriches policies with metadata for explainability
    """
    # Initialize optional parameters
    entities = entities or []
    comparison_targets = comparison_targets or []
    runtime_overrides = runtime_overrides or {}
    expected_answer_slots = expected_answer_slots or []
    
    # Safe fallback: use analytical as default for unknown types
    # Analytical provides balanced retrieval for uncertain cases
    if query_type not in QUERY_POLICY:
        query_type = "analytical"
    
    # Deep copy the base policy to prevent mutation of the original
    policy = copy.deepcopy(QUERY_POLICY[query_type])
    
    # Entity coverage adaptive policy
    # If query requires entity coverage, adapt retrieval parameters
    if requires_entity_coverage:
        if query_type == "factual":
            # Factual queries with entity coverage need deeper retrieval
            policy["retrieval_top_k"] = 8
            policy["rerank_top_n"] = 6
            policy["context_top_n"] = 3
            policy["use_context_selector"] = True
            policy["use_reranker"] = True
            policy["max_new_tokens"] = 50
            policy["evidence_shape"] = "multi_slot_evidence"
        elif query_type == "comparative":
            # Comparative queries with entity coverage
            policy["retrieval_top_k"] = 10
            policy["rerank_top_n"] = 8
            policy["context_top_n"] = 4
            policy["evidence_shape"] = "balanced_evidence_for_each_target"
        elif query_type == "multi_hop":
            # Multi-hop queries with entity coverage
            policy["retrieval_top_k"] = 10
            policy["rerank_top_n"] = 8
            policy["context_top_n"] = 4
            policy["use_context_selector"] = True
            policy["evidence_shape"] = "complementary_multi_hop_evidence"
    
    # Dynamic adjustments based on query complexity
    # These adjustments maintain the policy's core intent while adapting to specific queries
    
    # Increase retrieval depth for queries with many entities
    if len(entities) > 3:
        policy["retrieval_top_k"] = min(policy["retrieval_top_k"] + 3, 25)
        policy["context_top_n"] = min(policy["context_top_n"] + 2, 12)
    
    # Increase retrieval for comparative queries with multiple targets
    # Ensures balanced evidence for all comparison entities
    if query_type == "comparative" and len(comparison_targets) > 2:
        policy["retrieval_top_k"] = min(policy["retrieval_top_k"] + 4, 25)
        policy["context_top_n"] = min(policy["context_top_n"] + 3, 12)
    
    # Apply runtime overrides if provided
    # This enables agent-level customization without modifying the core policy
    if runtime_overrides:
        for key, value in runtime_overrides.items():
            if key in policy:
                policy[key] = value
    
    # Enrich policy with metadata for explainability and analytics
    policy["metadata"] = {
        "complexity": _determine_complexity(query_type),
        "reasoning_required": _requires_reasoning(query_type),
        "expected_latency": _estimate_latency(query_type),
        "retrieval_mode": "adaptive",
        "policy_source": query_type,
        "requires_entity_coverage": requires_entity_coverage,
        "expected_answer_slots": expected_answer_slots,
    }
    
    return policy


def _determine_complexity(query_type: str) -> str:
    """
    Determine the complexity level of a query type.
    
    Args:
        query_type: The query type string
        
    Returns:
        Complexity level: "low", "medium", or "high"
    """
    low_complexity = ["factual", "unanswerable"]
    medium_complexity = ["reasoning", "comparative", "temporal", "procedural"]
    high_complexity = ["analytical", "multi_hop"]
    
    if query_type in low_complexity:
        return "low"
    elif query_type in medium_complexity:
        return "medium"
    elif query_type in high_complexity:
        return "high"
    else:
        return "medium"  # Default fallback


def _requires_reasoning(query_type: str) -> bool:
    """
    Determine if a query type requires reasoning capabilities.
    
    Args:
        query_type: The query type string
        
    Returns:
        True if reasoning is required, False otherwise
    """
    reasoning_types = ["reasoning", "analytical", "multi_hop"]
    return query_type in reasoning_types


def _estimate_latency(query_type: str) -> str:
    """
    Estimate the expected latency for a query type.
    
    Args:
        query_type: The query type string
        
    Returns:
        Latency estimate: "low", "medium", or "high"
    """
    low_latency = ["factual", "unanswerable"]
    medium_latency = ["reasoning", "procedural"]
    high_latency = ["comparative", "temporal", "analytical", "multi_hop"]
    
    if query_type in low_latency:
        return "low"
    elif query_type in medium_latency:
        return "medium"
    elif query_type in high_latency:
        return "high"
    else:
        return "medium"  # Default fallback


__all__ = [
    'QUERY_POLICY',
    'build_retrieval_policy',
]
