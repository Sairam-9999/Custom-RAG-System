"""
Query Decomposition Module

This module provides evidence planning and decomposition planning for complex queries.
"""

from typing import Dict, Any, List

from ..core.types import QueryAnalysis


def build_evidence_requirements(analysis: QueryAnalysis) -> Dict[str, Any]:
    """
    Build evidence requirements based on query analysis.
    
    This function determines what evidence the retriever must collect
    to answer the user's query effectively.
    
    Args:
        analysis: QueryAnalysis object with parsed query information
        
    Returns:
        Dictionary of evidence requirements
    """
    query_type = analysis.query_type
    requirements = {
        "needs_single_fact": False,
        "needs_multiple_chunks": False,
        "needs_balanced_retrieval": False,
        "needs_temporal_ordering": False,
        "needs_step_sequence": False,
        "needs_interpretive_context": False,
        "needs_multi_hop_bridge": False,
        "needs_entity_linking": False,
        "needs_causal_support": False,
        "needs_comparison_axes": False,
        "needs_clarification": False,
        "preferred_evidence_shape": "standard",
    }
    
    # Evidence requirements by query type
    if query_type == "factual":
        requirements.update({
            "needs_single_fact": True,
            "needs_multiple_chunks": False,
            "preferred_evidence_shape": "single_supporting_sentence",
        })
    
    elif query_type == "reasoning":
        requirements.update({
            "needs_multiple_chunks": True,
            "needs_causal_support": True,
            "preferred_evidence_shape": "cause_effect_context",
        })
    
    elif query_type == "comparative":
        requirements.update({
            "needs_multiple_chunks": True,
            "needs_balanced_retrieval": True,
            "needs_comparison_axes": True,
            "preferred_evidence_shape": "balanced_evidence_for_each_target",
        })
    
    elif query_type == "temporal":
        requirements.update({
            "needs_multiple_chunks": True,
            "needs_temporal_ordering": True,
            "preferred_evidence_shape": "chronological_sequence",
        })
    
    elif query_type == "procedural":
        requirements.update({
            "needs_multiple_chunks": True,
            "needs_step_sequence": True,
            "preferred_evidence_shape": "ordered_steps",
        })
    
    elif query_type == "analytical":
        requirements.update({
            "needs_multiple_chunks": True,
            "needs_interpretive_context": True,
            "preferred_evidence_shape": "broad_context_with_supporting_clues",
        })
    
    elif query_type == "multi_hop":
        requirements.update({
            "needs_multiple_chunks": True,
            "needs_multi_hop_bridge": True,
            "needs_entity_linking": True,
            "preferred_evidence_shape": "linked_facts_across_chunks",
        })
    
    elif query_type == "unanswerable":
        requirements.update({
            "needs_clarification": True,
            "preferred_evidence_shape": "insufficient_or_ambiguous",
        })
    
    # Adjust based on parsed features
    if len(analysis.time_words) > 0:
        requirements["needs_temporal_ordering"] = True
    
    if len(analysis.comparison_targets) > 0:
        requirements["needs_balanced_retrieval"] = True
        requirements["needs_comparison_axes"] = True
    
    if len(analysis.entities) >= 2 and query_type != "comparative":
        requirements["needs_entity_linking"] = True
    
    if analysis.ambiguity_flags:
        requirements["needs_clarification"] = True
    
    return requirements


def build_decomposition_plan(analysis: QueryAnalysis) -> List[str]:
    """
    Build decomposition plan for complex queries.
    
    This function breaks down complex queries into sub-tasks
    that can be executed sequentially or in parallel.
    
    Args:
        analysis: QueryAnalysis object with parsed query information
        
    Returns:
        List of decomposition steps
    """
    query_type = analysis.query_type
    plan = []
    
    # Simple factual queries need no decomposition
    if query_type == "factual":
        return plan
    
    # Comparative queries: retrieve evidence for each target
    if query_type == "comparative":
        for target in analysis.comparison_targets:
            plan.append(f"Retrieve evidence about {target}")
        if analysis.question_focus:
            plan.append(f"Compare both targets using the question focus: {analysis.question_focus}")
        else:
            plan.append("Compare both targets across relevant dimensions")
    
    # Temporal queries: identify and order events
    elif query_type == "temporal":
        plan.append("Retrieve events related to the main subject")
        if analysis.time_words:
            plan.append(f"Identify temporal markers: {', '.join(analysis.time_words)}")
        plan.append("Order evidence chronologically")
    
    # Procedural queries: identify steps
    elif query_type == "procedural":
        plan.append("Retrieve procedural instructions and steps")
        if analysis.question_focus:
            plan.append(f"Focus on: {analysis.question_focus}")
        plan.append("Order steps in logical sequence")
    
    # Analytical queries: gather context and interpretive clues
    elif query_type == "analytical":
        plan.append("Retrieve broad context about the subject")
        if analysis.question_focus:
            plan.append(f"Focus analysis on: {analysis.question_focus}")
        plan.append("Gather supporting clues and evidence")
    
    # Multi-hop queries: connect entities across chunks
    elif query_type == "multi_hop":
        plan.append("Identify entity relationships")
        for entity in analysis.entities:
            plan.append(f"Retrieve evidence for entity: {entity}")
        plan.append("Find bridge facts connecting the entities")
        plan.append("Synthesize final answer only from connected evidence")
    
    # Unanswerable queries: request clarification
    elif query_type == "unanswerable":
        plan.append("Request clarification from user")
        if analysis.ambiguity_flags:
            plan.append(f"Address ambiguity flags: {', '.join(analysis.ambiguity_flags)}")
    
    return plan


__all__ = [
    'build_evidence_requirements',
    'build_decomposition_plan',
]
