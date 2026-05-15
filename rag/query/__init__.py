"""
Query Understanding Module

This module provides the main orchestration for query understanding,
integrating classification, parsing, slot extraction, decomposition, and policy building.
"""

from .classifier import (
    classify_query,
    get_query_type_description,
    QUERY_TYPES,
    PRIORITY_ORDER,
)
from .parser import (
    extract_entities,
    extract_keywords,
    extract_temporal_markers,
    extract_time_words,
    extract_comparison_targets,
    extract_question_focus,
    detect_ambiguity,
)
from .slot_extractor import (
    detect_entity_coverage_need,
    normalize_anchor_entity,
    extract_expected_answer_slots,
    build_answer_targets,
    extract_slot_answers_from_evidence,
)
from .decomposition import (
    build_evidence_requirements,
    build_decomposition_plan,
)
from .policy_engine import (
    QUERY_POLICY,
    build_retrieval_policy,
)

from ..core.types import QueryAnalysis, AnswerTarget


def understand_query(query: str, debug: bool = False) -> QueryAnalysis:
    """
    Main entry point for query understanding using the rule-based classification engine.
    
    This function orchestrates the complete query analysis pipeline:
    1. Classify query type using rule-based engine
    2. Parse entities and keywords
    3. Parse specialized markers (temporal, comparison, etc.)
    4. Extract question focus
    5. Detect ambiguity
    6. Determine retrieval requirements
    7. Extract answer targets for entity coverage
    8. Build retrieval policy
    9. Build evidence requirements
    10. Build decomposition plan
    11. Return QueryAnalysis with all parsing and planning results
    
    Args:
        query: The user query string
        debug: If True, include detailed debug metadata in the analysis
        
    Returns:
        QueryAnalysis object with complete query intelligence
        
    Example:
        >>> analysis = understand_query("Compare BM25 vs dense retrieval for factual QA")
        >>> print(analysis.query_type)
        'comparative'
        >>> print(analysis.question_focus)
        'factual qa'
        >>> print(analysis.evidence_requirements['needs_balanced_retrieval'])
        True
    """
    # Step 1: Extract entities and keywords
    entities = extract_entities(query)
    keywords = extract_keywords(query)
    
    # Step 2: Detect ambiguity
    ambiguity_flags = detect_ambiguity(query)
    
    # Step 3: Classify query using rule-based engine
    query_type, confidence, classification_metadata = classify_query(query, entities, ambiguity_flags, debug=True)
    
    # Step 4: Extract specialized markers
    comparison_targets = extract_comparison_targets(query)
    time_words = extract_time_words(query)
    
    # Step 5: Extract question focus
    question_focus = extract_question_focus(query, query_type)
    
    # Step 6: Determine retrieval requirements
    requires_multiple_chunks = query_type in ['comparative', 'temporal', 'procedural', 'analytical', 'multi_hop']
    needs_reranker = query_type in ['reasoning', 'comparative', 'temporal', 'procedural', 'analytical', 'multi_hop', 'unanswerable']
    needs_context_selector = query_type in ['reasoning', 'comparative', 'temporal', 'procedural', 'analytical', 'multi_hop']
    
    requires_reasoning = query_type in ['reasoning', 'analytical', 'multi_hop']
    requires_multi_hop = query_type == 'multi_hop' or (len(entities) >= 2 and query_type != 'comparative')
    requires_temporal_ordering = query_type == 'temporal' or len(time_words) > 0
    
    # Step 7: Detect entity coverage need and extract expected answer slots
    requires_entity_coverage = detect_entity_coverage_need(query, query_type, entities)
    anchor_entity = None
    expected_answer_slots = []
    answer_targets = []
    if requires_entity_coverage:
        anchor_entity, expected_answer_slots = extract_expected_answer_slots(query, query_type, entities)
        # Build semantic answer targets
        anchor_entity, answer_targets = build_answer_targets(query, query_type, entities)
    
    # Step 8: Build retrieval policy with entity coverage parameters
    retrieval_strategy = build_retrieval_policy(
        query_type=query_type,
        entities=entities,
        comparison_targets=comparison_targets,
        requires_entity_coverage=requires_entity_coverage,
        expected_answer_slots=expected_answer_slots,
    )
    
    # Step 9: Determine answer style
    answer_style = retrieval_strategy.get('answer_style', 'default')
    
    # Step 10: Build partial analysis for evidence planning
    partial_analysis = QueryAnalysis(
        query=query,
        query_type=query_type,
        confidence=confidence,
        entities=entities,
        keywords=keywords,
        comparison_targets=comparison_targets,
        time_words=time_words,
        question_focus=question_focus,
        ambiguity_flags=ambiguity_flags,
        requires_multiple_chunks=requires_multiple_chunks,
        needs_reranker=needs_reranker,
        needs_context_selector=needs_context_selector,
        requires_reasoning=requires_reasoning,
        requires_multi_hop=requires_multi_hop,
        requires_temporal_ordering=requires_temporal_ordering,
        requires_entity_coverage=requires_entity_coverage,
        anchor_entity=anchor_entity,
        expected_answer_slots=expected_answer_slots,
        answer_targets=answer_targets,
        retrieval_strategy=retrieval_strategy,
        answer_style=answer_style,
    )
    
    # Step 11: Build evidence requirements
    evidence_requirements = build_evidence_requirements(partial_analysis)
    
    # Step 12: Build decomposition plan
    decomposition_plan = build_decomposition_plan(partial_analysis)
    
    # Step 13: Enrich retrieval strategy with evidence planning
    retrieval_strategy['evidence_shape'] = evidence_requirements.get('preferred_evidence_shape', 'standard')
    retrieval_strategy['needs_balanced_retrieval'] = evidence_requirements.get('needs_balanced_retrieval', False)
    
    # Step 14: Enrich metadata with classification and debug information
    metadata = {
        'query_length': len(query),
        'word_count': len(query.split()),
        'entity_count': len(entities),
        'keyword_count': len(keywords),
        'has_comparison_markers': len(comparison_targets) > 0,
        'has_temporal_markers': len(time_words) > 0,
        'has_ambiguity': len(ambiguity_flags) > 0,
    }
    
    # Add classification metadata for explainability
    if debug:
        # Add parsing metadata
        parsing_metadata = {
            'parsing': {
                'entities_source': 'capitalized_phrases_quotes_technical_terms',
                'time_words_found': time_words,
                'comparison_pattern': 'detected' if comparison_targets else 'none',
                'focus_extraction_rule': 'query_type_specific',
            },
            'planning': {
                'evidence_shape': evidence_requirements.get('preferred_evidence_shape'),
                'decomposition_required': len(decomposition_plan) > 0,
                'planning_reason': f'{query_type}_query_requires_specific_evidence',
            },
        }
        metadata.update(classification_metadata)
        metadata.update(parsing_metadata)
    else:
        # Always include basic classification metadata
        metadata['matched_rules'] = classification_metadata.get('matched_rules', [])
        metadata['reason'] = classification_metadata.get('reason', '')
    
    # Construct and return final analysis
    analysis = QueryAnalysis(
        query=query,
        query_type=query_type,
        confidence=confidence,
        entities=entities,
        keywords=keywords,
        comparison_targets=comparison_targets,
        time_words=time_words,
        question_focus=question_focus,
        ambiguity_flags=ambiguity_flags,
        requires_multiple_chunks=requires_multiple_chunks,
        needs_reranker=needs_reranker,
        needs_context_selector=needs_context_selector,
        requires_reasoning=requires_reasoning,
        requires_multi_hop=requires_multi_hop,
        requires_temporal_ordering=requires_temporal_ordering,
        requires_entity_coverage=requires_entity_coverage,
        anchor_entity=anchor_entity,
        expected_answer_slots=expected_answer_slots,
        answer_targets=answer_targets,
        retrieval_strategy=retrieval_strategy,
        answer_style=answer_style,
        evidence_requirements=evidence_requirements,
        decomposition_plan=decomposition_plan,
        metadata=metadata,
    )
    
    return analysis


def print_query_analysis(analysis: QueryAnalysis, verbose: bool = False) -> None:
    """
    Pretty-print query analysis for debugging and inspection.
    
    Args:
        analysis: QueryAnalysis object to print
        verbose: If True, print detailed debug metadata
    """
    print("=" * 70)
    print("QUERY ANALYSIS")
    print("=" * 70)
    print(f"Query: {analysis.query}")
    print(f"Type: {analysis.query_type}")
    print(f"Confidence: {analysis.confidence:.2f}")
    print()
    print("Extracted Information:")
    print(f"  Entities: {analysis.entities}")
    print(f"  Keywords: {analysis.keywords}")
    print(f"  Comparison Targets: {analysis.comparison_targets}")
    print(f"  Time Words: {analysis.time_words}")
    print(f"  Question Focus: {analysis.question_focus}")
    print(f"  Ambiguity Flags: {analysis.ambiguity_flags}")
    print()
    print("Retrieval Requirements:")
    print(f"  Requires Multiple Chunks: {analysis.requires_multiple_chunks}")
    print(f"  Needs Reranker: {analysis.needs_reranker}")
    print(f"  Needs Context Selector: {analysis.needs_context_selector}")
    print(f"  Requires Reasoning: {analysis.requires_reasoning}")
    print(f"  Requires Multi-Hop: {analysis.requires_multi_hop}")
    print(f"  Requires Temporal Ordering: {analysis.requires_temporal_ordering}")
    print()
    print("Evidence Requirements:")
    for key, value in analysis.evidence_requirements.items():
        print(f"  {key}: {value}")
    print()
    if analysis.decomposition_plan:
        print("Decomposition Plan:")
        for i, step in enumerate(analysis.decomposition_plan, 1):
            print(f"  {i}. {step}")
        print()
    print("Retrieval Strategy:")
    for key, value in analysis.retrieval_strategy.items():
        print(f"  {key}: {value}")
    print()
    print(f"Answer Style: {analysis.answer_style}")
    print()
    print("Classification Metadata:")
    if 'matched_rules' in analysis.metadata:
        print(f"  Matched Rules: {analysis.metadata['matched_rules']}")
    if 'reason' in analysis.metadata:
        print(f"  Reason: {analysis.metadata['reason']}")
    if verbose:
        if 'parsing' in analysis.metadata:
            print("  Parsing Metadata:")
            for key, value in analysis.metadata['parsing'].items():
                print(f"    {key}: {value}")
        if 'planning' in analysis.metadata:
            print("  Planning Metadata:")
            for key, value in analysis.metadata['planning'].items():
                print(f"    {key}: {value}")
        if 'rule_scores' in analysis.metadata:
            print(f"  Rule Scores: {analysis.metadata['rule_scores']}")
        if 'matched_markers' in analysis.metadata:
            print(f"  Matched Markers: {analysis.metadata['matched_markers']}")
        if 'conflicting_types' in analysis.metadata:
            print(f"  Conflicting Types: {analysis.metadata['conflicting_types']}")
        if 'priority_winner' in analysis.metadata:
            print(f"  Priority Winner: {analysis.metadata['priority_winner']}")
    print()
    print("Query Statistics:")
    print(f"  Query Length: {analysis.metadata.get('query_length', 0)}")
    print(f"  Word Count: {analysis.metadata.get('word_count', 0)}")
    print(f"  Entity Count: {analysis.metadata.get('entity_count', 0)}")
    print(f"  Keyword Count: {analysis.metadata.get('keyword_count', 0)}")
    print("=" * 70)


__all__ = [
    # Main orchestration
    'understand_query',
    'print_query_analysis',
    # Classifier
    'classify_query',
    'get_query_type_description',
    'QUERY_TYPES',
    'PRIORITY_ORDER',
    # Parser
    'extract_entities',
    'extract_keywords',
    'extract_temporal_markers',
    'extract_time_words',
    'extract_comparison_targets',
    'extract_question_focus',
    'detect_ambiguity',
    # Slot extractor
    'detect_entity_coverage_need',
    'normalize_anchor_entity',
    'extract_expected_answer_slots',
    'build_answer_targets',
    'extract_slot_answers_from_evidence',
    # Decomposition
    'build_evidence_requirements',
    'build_decomposition_plan',
    # Policy engine
    'QUERY_POLICY',
    'build_retrieval_policy',
]
