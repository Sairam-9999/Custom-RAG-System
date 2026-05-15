"""
Retrieval Retry and Self-Correction Loop for Adaptive RAG System - Phase 9

This module provides a self-correcting retrieval system that automatically
retries retrieval and regeneration when answer validation fails or evidence
is insufficient.

Architecture:
    Retrieve → Generate → Validate → Detect Failure → Retry Retrieval →
    Regenerate → Revalidate → Return Best Grounded Answer

The system is:
- Validation-driven: Uses validation results to decide retry necessity
- Query-type-aware: Adapts retry strategy based on query classification
- Evidence-focused: Prioritizes retrieving better evidence over regenerating
- Safety-first: Never infinitely retries, amplifies hallucinations, or keeps unsupported answers
- Modular: Generator-agnostic, retrieval-aware, validation-aware
- Extensible: Supports future agentic retrieval and planner agents
- Production-oriented: Deterministic heuristics, lightweight retry logic, explainable decisions
"""

import re
import copy
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class RetryDecision:
    """
    Decision object for whether and how to retry retrieval.
    
    This dataclass captures all information needed to execute a retry,
    including the reason for retry, the strategy to use, and any
    modified parameters for the retry attempt.
    
    Attributes:
        should_retry: Whether a retry should be attempted
        retry_reason: The reason for the retry (e.g., "low_evidence_coverage")
        retry_strategy: The retry strategy to use (e.g., "expand_retrieval")
        retry_query: The query to use for retry (possibly reformulated)
        expanded_top_k: The expanded retrieval top_k for retry
        use_deeper_reranking: Whether to use deeper reranking on retry
        use_broader_context: Whether to use broader context on retry
        max_retry_attempts: Maximum number of retry attempts (default 2)
        metadata: Additional metadata about the retry decision
    """
    should_retry: bool
    retry_reason: str
    retry_strategy: str
    retry_query: str
    expanded_top_k: int
    use_deeper_reranking: bool
    use_broader_context: bool
    max_retry_attempts: int = 2
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetryAttempt:
    """
    Record of a single retry attempt.
    
    This dataclass stores the results of a single retrieval/generation
    attempt for later comparison and best answer selection.
    
    Attributes:
        attempt_number: The attempt number (1-based)
        query: The query used for this attempt
        answer: The generated answer
        validation: The validation result for this answer
        retrieval_policy: The retrieval policy used
        timings: Timing information for this attempt
        metadata: Additional metadata about this attempt
    """
    attempt_number: int
    query: str
    answer: str
    validation: Any
    retrieval_policy: Dict[str, Any]
    timings: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# RETRY STRATEGY DEFINITIONS
# ============================================================================

RETRY_STRATEGIES = [
    "expand_retrieval",
    "reformulate_query",
    "broaden_context",
    "increase_rerank_depth",
    "decompose_query",
    "hybrid_retry",
    "no_retry",
]


# ============================================================================
# RETRY DECISION LOGIC
# ============================================================================

def decide_retry(
    query: str,
    answer: str,
    validation: Any,
    query_analysis: Optional[Any] = None,
    retrieval_results: Optional[List[Any]] = None,
) -> RetryDecision:
    """
    Decide whether to retry retrieval based on validation results.
    
    This function analyzes the validation result to determine if a retry
    is warranted, and if so, which retry strategy to use. The decision
    is query-type-aware, adapting the retry strategy based on the query
    classification.
    
    Args:
        query: The original user query
        answer: The generated answer
        validation: The AnswerValidationResult from validate_answer()
        query_analysis: Optional QueryAnalysis object for query-type awareness
        retrieval_results: Optional retrieval results for evidence analysis
        
    Returns:
        RetryDecision with retry instructions
        
    Example:
        >>> decision = decide_retry(
        ...     query="What is X?",
        ...     answer="X is Y.",
        ...     validation=validation_result,
        ...     query_analysis=analysis,
        ... )
        >>> if decision.should_retry:
        ...     execute_retry(decision)
    """
    # Extract validation metrics
    grounded = getattr(validation, 'grounded', False)
    confidence = getattr(validation, 'confidence', 0.0)
    validation_action = getattr(validation, 'validation_action', 'accept')
    hallucination_detected = getattr(validation, 'hallucination_detected', False)
    evidence_coverage = getattr(validation, 'evidence_coverage', 0.0)
    contradiction_count = len(getattr(validation, 'contradictions', []))
    unsupported_claims_count = len(getattr(validation, 'unsupported_claims', []))
    missing_entities_count = len(getattr(validation, 'missing_entities', []))
    
    # Get query type for query-type-aware retries
    query_type = "factual"  # Default
    if query_analysis is not None:
        query_type = getattr(query_analysis, 'query_type', 'factual')
    
    # DO NOT RETRY if answer is well-grounded
    if grounded and confidence >= 0.7 and validation_action == "accept" and evidence_coverage >= 0.7:
        return RetryDecision(
            should_retry=False,
            retry_reason="answer_well_grounded",
            retry_strategy="no_retry",
            retry_query=query,
            expanded_top_k=0,
            use_deeper_reranking=False,
            use_broader_context=False,
            metadata={
                "confidence": confidence,
                "evidence_coverage": evidence_coverage,
                "validation_action": validation_action,
            },
        )
    
    # RETRY if validation action indicates revision or refusal
    if validation_action in ["revise", "refuse"]:
        return _build_retry_decision_for_action(
            query=query,
            query_type=query_type,
            validation_action=validation_action,
            evidence_coverage=evidence_coverage,
            hallucination_detected=hallucination_detected,
            contradiction_count=contradiction_count,
            unsupported_claims_count=unsupported_claims_count,
            missing_entities_count=missing_entities_count,
        )
    
    # RETRY if hallucination detected
    if hallucination_detected:
        return _build_retry_decision_for_hallucination(
            query=query,
            query_type=query_type,
            missing_entities_count=missing_entities_count,
            unsupported_claims_count=unsupported_claims_count,
        )
    
    # RETRY if evidence coverage too low
    if evidence_coverage < 0.5:
        return _build_retry_decision_for_low_evidence(
            query=query,
            query_type=query_type,
            evidence_coverage=evidence_coverage,
        )
    
    # RETRY if contradictions detected
    if contradiction_count >= 1:
        return _build_retry_decision_for_contradiction(
            query=query,
            query_type=query_type,
            contradiction_count=contradiction_count,
        )
    
    # RETRY if confidence too low
    if confidence < 0.5:
        return _build_retry_decision_for_low_confidence(
            query=query,
            query_type=query_type,
            confidence=confidence,
            evidence_coverage=evidence_coverage,
        )
    
    # Default: no retry
    return RetryDecision(
        should_retry=False,
        retry_reason="no_clear_failure",
        retry_strategy="no_retry",
        retry_query=query,
        expanded_top_k=0,
        use_deeper_reranking=False,
        use_broader_context=False,
        metadata={
            "confidence": confidence,
            "evidence_coverage": evidence_coverage,
        },
    )


def _build_retry_decision_for_action(
    query: str,
    query_type: str,
    validation_action: str,
    evidence_coverage: float,
    hallucination_detected: bool,
    contradiction_count: int,
    unsupported_claims_count: int,
    missing_entities_count: int,
) -> RetryDecision:
    """
    Build retry decision when validation action is revise or refuse.
    """
    if query_type == "factual":
        # Factual: slightly expand retrieval, prioritize precision
        return RetryDecision(
            should_retry=True,
            retry_reason=f"validation_action_{validation_action}",
            retry_strategy="expand_retrieval",
            retry_query=query,
            expanded_top_k=5,
            use_deeper_reranking=True,
            use_broader_context=False,
            metadata={
                "validation_action": validation_action,
                "evidence_coverage": evidence_coverage,
                "query_type": query_type,
            },
        )
    elif query_type == "multi_hop":
        # Multi-hop: significantly expand retrieval, increase rerank depth
        return RetryDecision(
            should_retry=True,
            retry_reason=f"validation_action_{validation_action}",
            retry_strategy="expand_retrieval",
            retry_query=query,
            expanded_top_k=10,
            use_deeper_reranking=True,
            use_broader_context=True,
            metadata={
                "validation_action": validation_action,
                "evidence_coverage": evidence_coverage,
                "query_type": query_type,
            },
        )
    elif query_type == "comparative":
        # Comparative: ensure balanced evidence
        return RetryDecision(
            should_retry=True,
            retry_reason=f"validation_action_{validation_action}",
            retry_strategy="expand_retrieval",
            retry_query=query,
            expanded_top_k=8,
            use_deeper_reranking=True,
            use_broader_context=True,
            metadata={
                "validation_action": validation_action,
                "evidence_coverage": evidence_coverage,
                "query_type": query_type,
            },
        )
    elif query_type == "temporal":
        # Temporal: retrieve surrounding timeline context
        return RetryDecision(
            should_retry=True,
            retry_reason=f"validation_action_{validation_action}",
            retry_strategy="broaden_context",
            retry_query=query,
            expanded_top_k=8,
            use_deeper_reranking=False,
            use_broader_context=True,
            metadata={
                "validation_action": validation_action,
                "evidence_coverage": evidence_coverage,
                "query_type": query_type,
            },
        )
    elif query_type == "unanswerable":
        # Unanswerable: minimal retries, prioritize refusal safety
        return RetryDecision(
            should_retry=True,
            retry_reason=f"validation_action_{validation_action}",
            retry_strategy="expand_retrieval",
            retry_query=query,
            expanded_top_k=3,
            use_deeper_reranking=False,
            use_broader_context=False,
            max_retry_attempts=1,  # Only retry once for unanswerable
            metadata={
                "validation_action": validation_action,
                "evidence_coverage": evidence_coverage,
                "query_type": query_type,
            },
        )
    else:
        # Default: moderate expansion
        return RetryDecision(
            should_retry=True,
            retry_reason=f"validation_action_{validation_action}",
            retry_strategy="expand_retrieval",
            retry_query=query,
            expanded_top_k=5,
            use_deeper_reranking=True,
            use_broader_context=False,
            metadata={
                "validation_action": validation_action,
                "evidence_coverage": evidence_coverage,
                "query_type": query_type,
            },
        )


def _build_retry_decision_for_hallucination(
    query: str,
    query_type: str,
    missing_entities_count: int,
    unsupported_claims_count: int,
) -> RetryDecision:
    """
    Build retry decision when hallucination is detected.
    """
    if query_type == "factual":
        # Factual: reformulate query to be more specific
        reformulated = reformulate_query(query, None)
        return RetryDecision(
            should_retry=True,
            retry_reason="hallucination_detected",
            retry_strategy="reformulate_query" if reformulated != query else "expand_retrieval",
            retry_query=reformulated,
            expanded_top_k=5,
            use_deeper_reranking=True,
            use_broader_context=False,
            metadata={
                "missing_entities_count": missing_entities_count,
                "unsupported_claims_count": unsupported_claims_count,
                "query_type": query_type,
            },
        )
    elif query_type == "multi_hop":
        # Multi-hop: decompose query and retrieve for each subquery
        return RetryDecision(
            should_retry=True,
            retry_reason="hallucination_detected",
            retry_strategy="decompose_query",
            retry_query=query,
            expanded_top_k=10,
            use_deeper_reranking=True,
            use_broader_context=True,
            metadata={
                "missing_entities_count": missing_entities_count,
                "unsupported_claims_count": unsupported_claims_count,
                "query_type": query_type,
            },
        )
    else:
        # Default: expand retrieval
        return RetryDecision(
            should_retry=True,
            retry_reason="hallucination_detected",
            retry_strategy="expand_retrieval",
            retry_query=query,
            expanded_top_k=8,
            use_deeper_reranking=True,
            use_broader_context=True,
            metadata={
                "missing_entities_count": missing_entities_count,
                "unsupported_claims_count": unsupported_claims_count,
                "query_type": query_type,
            },
        )


def _build_retry_decision_for_low_evidence(
    query: str,
    query_type: str,
    evidence_coverage: float,
) -> RetryDecision:
    """
    Build retry decision when evidence coverage is low.
    """
    if query_type == "factual":
        # Factual: slightly expand top_k
        return RetryDecision(
            should_retry=True,
            retry_reason="low_evidence_coverage",
            retry_strategy="expand_retrieval",
            retry_query=query,
            expanded_top_k=5,
            use_deeper_reranking=True,
            use_broader_context=False,
            metadata={
                "evidence_coverage": evidence_coverage,
                "query_type": query_type,
            },
        )
    elif query_type == "multi_hop":
        # Multi-hop: significantly expand retrieval
        return RetryDecision(
            should_retry=True,
            retry_reason="low_evidence_coverage",
            retry_strategy="expand_retrieval",
            retry_query=query,
            expanded_top_k=10,
            use_deeper_reranking=True,
            use_broader_context=True,
            metadata={
                "evidence_coverage": evidence_coverage,
                "query_type": query_type,
            },
        )
    elif query_type == "comparative":
        # Comparative: ensure balanced evidence
        return RetryDecision(
            should_retry=True,
            retry_reason="low_evidence_coverage",
            retry_strategy="expand_retrieval",
            retry_query=query,
            expanded_top_k=8,
            use_deeper_reranking=True,
            use_broader_context=True,
            metadata={
                "evidence_coverage": evidence_coverage,
                "query_type": query_type,
            },
        )
    else:
        # Default: moderate expansion
        return RetryDecision(
            should_retry=True,
            retry_reason="low_evidence_coverage",
            retry_strategy="expand_retrieval",
            retry_query=query,
            expanded_top_k=6,
            use_deeper_reranking=True,
            use_broader_context=False,
            metadata={
                "evidence_coverage": evidence_coverage,
                "query_type": query_type,
            },
        )


def _build_retry_decision_for_contradiction(
    query: str,
    query_type: str,
    contradiction_count: int,
) -> RetryDecision:
    """
    Build retry decision when contradictions are detected.
    """
    # Contradictions suggest wrong evidence was retrieved
    # Reformulate query to be more specific
    reformulated = reformulate_query(query, None)
    
    if query_type == "factual":
        return RetryDecision(
            should_retry=True,
            retry_reason="contradiction_detected",
            retry_strategy="reformulate_query" if reformulated != query else "expand_retrieval",
            retry_query=reformulated,
            expanded_top_k=6,
            use_deeper_reranking=True,
            use_broader_context=False,
            metadata={
                "contradiction_count": contradiction_count,
                "query_type": query_type,
            },
        )
    else:
        return RetryDecision(
            should_retry=True,
            retry_reason="contradiction_detected",
            retry_strategy="expand_retrieval",
            retry_query=reformulated,
            expanded_top_k=8,
            use_deeper_reranking=True,
            use_broader_context=True,
            metadata={
                "contradiction_count": contradiction_count,
                "query_type": query_type,
            },
        )


def _build_retry_decision_for_low_confidence(
    query: str,
    query_type: str,
    confidence: float,
    evidence_coverage: float,
) -> RetryDecision:
    """
    Build retry decision when confidence is low.
    """
    if evidence_coverage < 0.5:
        # Low confidence due to low evidence - expand retrieval
        return RetryDecision(
            should_retry=True,
            retry_reason="low_confidence_low_evidence",
            retry_strategy="expand_retrieval",
            retry_query=query,
            expanded_top_k=6,
            use_deeper_reranking=True,
            use_broader_context=False,
            metadata={
                "confidence": confidence,
                "evidence_coverage": evidence_coverage,
                "query_type": query_type,
            },
        )
    else:
        # Low confidence despite decent evidence - might need query reformulation
        reformulated = reformulate_query(query, None)
        return RetryDecision(
            should_retry=True,
            retry_reason="low_confidence",
            retry_strategy="reformulate_query" if reformulated != query else "no_retry",
            retry_query=reformulated,
            expanded_top_k=0,
            use_deeper_reranking=False,
            use_broader_context=False,
            metadata={
                "confidence": confidence,
                "evidence_coverage": evidence_coverage,
                "query_type": query_type,
            },
        )


# ============================================================================
# QUERY REFORMULATION
# ============================================================================

def reformulate_query(query: str, query_analysis: Optional[Any] = None) -> str:
    """
    Reformulate a query to improve retrieval.
    
    This function attempts to improve the query by:
    - Removing ambiguity
    - Expanding entity references
    - Simplifying wording
    - Extracting core question
    - Adding missing keywords
    
    Args:
        query: The original query
        query_analysis: Optional QueryAnalysis object for additional context
        
    Returns:
        Reformulated query string (may be the same as original if no improvements found)
        
    Example:
        >>> reformulated = reformulate_query("What did he do?")
        >>> # Returns: "What did the main character do?" (if context available)
    """
    reformulated = query
    query_lower = query.lower()
    
    # Strategy 1: Remove vague pronouns if entities are available
    if query_analysis is not None:
        entities = getattr(query_analysis, 'entities', [])
        if entities:
            # Replace common pronouns with first entity
            pronoun_map = {
                'he': entities[0],
                'she': entities[0],
                'it': entities[0],
                'they': entities[0] if len(entities) > 1 else entities[0],
            }
            for pronoun, replacement in pronoun_map.items():
                if f' {pronoun} ' in f' {query_lower} ':
                    reformulated = re.sub(rf'\b{pronoun}\b', replacement, reformulated, flags=re.IGNORECASE)
                    break
    
    # Strategy 2: Expand abbreviations
    abbreviations = {
        'vs': 'versus',
        'etc': 'et cetera',
        'e.g.': 'for example',
        'i.e.': 'that is',
    }
    for abbrev, expansion in abbreviations.items():
        if abbrev in query_lower:
            reformulated = reformulated.replace(abbrev, expansion)
    
    # Strategy 3: Add question words if missing
    question_words = ['what', 'who', 'when', 'where', 'why', 'how', 'which']
    has_question_word = any(qw in query_lower for qw in question_words)
    
    if not has_question_word and not query.strip().endswith('?'):
        # Try to infer the question type
        if 'describe' in query_lower or 'explain' in query_lower:
            reformulated = 'What is ' + reformulated
        elif 'compare' in query_lower:
            reformulated = 'What is the difference between ' + reformulated.replace('compare', '')
        elif 'list' in query_lower:
            reformulated = 'What are ' + reformulated.replace('list', '')
    
    # Strategy 4: Simplify complex constructions
    # Remove excessive punctuation
    reformulated = re.sub(r'\s+', ' ', reformulated)  # Normalize whitespace
    reformulated = reformulated.strip()
    
    # Strategy 5: Extract core question if too verbose
    words = reformulated.split()
    if len(words) > 20:
        # Keep first 15 words + last 5 words (often contain the core question)
        core_start = ' '.join(words[:15])
        core_end = ' '.join(words[-5:])
        reformulated = f"{core_start} ... {core_end}"
    
    return reformulated


# ============================================================================
# QUERY DECOMPOSITION
# ============================================================================

def decompose_query(query: str) -> List[str]:
    """
    Decompose a multi-hop query into subqueries.
    
    This function breaks down complex multi-hop queries into simpler
    subqueries that can be answered independently and then combined.
    
    Args:
        query: The original multi-hop query
        
    Returns:
        List of subquery strings
        
    Example:
        >>> subqueries = decompose_query("Who warned X about Y when Z controlled the kingdom?")
        >>> # Returns: [
        >>> #     "Who warned X?",
        >>> #     "What was Y?",
        >>> #     "When did Z control the kingdom?",
        >>> # ]
    """
    subqueries = []
    query_lower = query.lower()
    
    # Pattern 1: "Who X when Y" -> ["Who X?", "When Y?"]
    who_when_pattern = r'who\s+(.+?)\s+when\s+(.+)'
    match = re.search(who_when_pattern, query_lower)
    if match:
        who_part = match.group(1).strip()
        when_part = match.group(2).strip()
        subqueries.append(f"Who {who_part}?")
        subqueries.append(f"When {when_part}?")
        return subqueries
    
    # Pattern 2: "What X when Y" -> ["What X?", "When Y?"]
    what_when_pattern = r'what\s+(.+?)\s+when\s+(.+)'
    match = re.search(what_when_pattern, query_lower)
    if match:
        what_part = match.group(1).strip()
        when_part = match.group(2).strip()
        subqueries.append(f"What {what_part}?")
        subqueries.append(f"When {when_part}?")
        return subqueries
    
    # Pattern 3: "Why X before Y" -> ["Why X?", "What happened before Y?"]
    why_before_pattern = r'why\s+(.+?)\s+before\s+(.+)'
    match = re.search(why_before_pattern, query_lower)
    if match:
        why_part = match.group(1).strip()
        before_part = match.group(2).strip()
        subqueries.append(f"Why {why_part}?")
        subqueries.append(f"What happened before {before_part}?")
        return subqueries
    
    # Pattern 4: "Why X after Y" -> ["Why X?", "What happened after Y?"]
    why_after_pattern = r'why\s+(.+?)\s+after\s+(.+)'
    match = re.search(why_after_pattern, query_lower)
    if match:
        why_part = match.group(1).strip()
        after_part = match.group(2).strip()
        subqueries.append(f"Why {why_part}?")
        subqueries.append(f"What happened after {after_part}?")
        return subqueries
    
    # Pattern 5: Multiple entities with temporal markers
    # Extract entities and temporal markers
    entities = _extract_entities_simple(query)
    temporal_markers = ['before', 'after', 'during', 'when', 'while']
    found_temporal = [tm for tm in temporal_markers if tm in query_lower]
    
    if len(entities) >= 2 and found_temporal:
        # Create subqueries for each entity
        for entity in entities:
            subqueries.append(f"What is {entity}?")
        # Add temporal context query
        temporal_query = f"What happened {' '.join(found_temporal[:2])}?"
        subqueries.append(temporal_query)
        return subqueries
    
    # Pattern 6: Chained clauses with "and"
    if ' and ' in query_lower:
        parts = query.split(' and ')
        if len(parts) >= 2:
            for part in parts:
                subqueries.append(part.strip() + '?')
            return subqueries
    
    # Default: return original as single subquery
    return [query]


def _extract_entities_simple(query: str) -> List[str]:
    """
    Simple entity extraction for query decomposition.
    
    Args:
        query: The query string
        
    Returns:
        List of extracted entities
    """
    entities = []
    
    # Extract capitalized phrases
    capitalized_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
    capitalized_matches = re.findall(capitalized_pattern, query)
    
    common_words = {'The', 'A', 'An', 'This', 'That', 'What', 'How', 'Why', 'Who'}
    for match in capitalized_matches:
        words = match.split()
        if words[0] not in common_words:
            entities.append(match)
    
    # Extract single-letter uppercase entities
    single_letter_pattern = r'\b[A-Z]\b'
    single_letter_matches = re.findall(single_letter_pattern, query)
    for match in single_letter_matches:
        if match not in common_words:
            entities.append(match)
    
    return entities


# ============================================================================
# RETRIEVAL POLICY EXPANSION
# ============================================================================

def expand_retrieval_policy(
    policy: Dict[str, Any],
    retry_strategy: str,
    retry_decision: Optional[RetryDecision] = None,
) -> Dict[str, Any]:
    """
    Expand retrieval policy based on retry strategy.
    
    This function modifies the retrieval policy parameters to implement
    the retry strategy, such as increasing top_k, rerank depth, or context size.
    
    Args:
        policy: The original retrieval policy dict
        retry_strategy: The retry strategy to apply
        retry_decision: Optional RetryDecision for specific parameters
        
    Returns:
        Expanded retrieval policy dict
        
    Example:
        >>> policy = {"retrieval_top_k": 10, "rerank_top_n": 5, "context_top_n": 3}
        >>> expanded = expand_retrieval_policy(policy, "expand_retrieval")
        >>> # Returns: {"retrieval_top_k": 15, "rerank_top_n": 8, "context_top_n": 5}
    """
    expanded = copy.deepcopy(policy)
    
    if retry_decision is not None and retry_decision.expanded_top_k > 0:
        # Use explicit parameters from retry decision
        expanded["retrieval_top_k"] += retry_decision.expanded_top_k
        if retry_decision.use_deeper_reranking:
            expanded["rerank_top_n"] = expanded.get("rerank_top_n", 5) + 3
        if retry_decision.use_broader_context:
            expanded["context_top_n"] = expanded.get("context_top_n", 3) + 2
        return expanded
    
    # Strategy-based expansion
    if retry_strategy == "expand_retrieval":
        expanded["retrieval_top_k"] += 5
        expanded["rerank_top_n"] = expanded.get("rerank_top_n", 5) + 3
        expanded["context_top_n"] = expanded.get("context_top_n", 3) + 2
    elif retry_strategy == "reformulate_query":
        # Query reformulation doesn't necessarily need policy expansion
        # Keep policy as-is
        pass
    elif retry_strategy == "broaden_context":
        expanded["context_top_n"] = expanded.get("context_top_n", 3) + 4
        expanded["retrieval_top_k"] += 3
    elif retry_strategy == "increase_rerank_depth":
        expanded["rerank_top_n"] = expanded.get("rerank_top_n", 5) + 5
        expanded["retrieval_top_k"] += 3
    elif retry_strategy == "decompose_query":
        # For decomposition, we need more retrieval for each subquery
        expanded["retrieval_top_k"] += 8
        expanded["rerank_top_n"] = expanded.get("rerank_top_n", 5) + 4
        expanded["context_top_n"] = expanded.get("context_top_n", 3) + 3
    elif retry_strategy == "hybrid_retry":
        # Aggressive expansion for hybrid retry
        expanded["retrieval_top_k"] += 10
        expanded["rerank_top_n"] = expanded.get("rerank_top_n", 5) + 5
        expanded["context_top_n"] = expanded.get("context_top_n", 3) + 4
    
    # Ensure minimum values
    expanded["retrieval_top_k"] = max(expanded["retrieval_top_k"], 5)
    expanded["rerank_top_n"] = max(expanded.get("rerank_top_n", 5), 3)
    expanded["context_top_n"] = max(expanded.get("context_top_n", 3), 2)
    
    return expanded


# ============================================================================
# BEST ANSWER SELECTION
# ============================================================================

def select_best_answer(attempts: List[RetryAttempt]) -> Tuple[str, RetryAttempt, Dict[str, Any]]:
    """
    Select the best answer from multiple retry attempts.
    
    This function compares all attempts and selects the one with the
    highest groundedness, confidence, and lowest hallucination risk.
    
    Args:
        attempts: List of RetryAttempt objects
        
    Returns:
        Tuple of (best_answer, best_attempt, selection_metadata)
        
    Example:
        >>> best_answer, best_attempt, metadata = select_best_answer(attempts)
        >>> return best_answer
    """
    if not attempts:
        return "", None, {"reason": "no_attempts"}
    
    if len(attempts) == 1:
        return attempts[0].answer, attempts[0], {"reason": "single_attempt"}
    
    # Score each attempt
    scored_attempts = []
    for attempt in attempts:
        validation = attempt.validation
        score = _compute_attempt_score(validation)
        scored_attempts.append((score, attempt))
    
    # Sort by score (descending)
    scored_attempts.sort(key=lambda x: x[0], reverse=True)
    
    # Select best
    best_score, best_attempt = scored_attempts[0]
    
    # Build selection metadata
    selection_metadata = {
        "total_attempts": len(attempts),
        "best_attempt_number": best_attempt.attempt_number,
        "best_score": best_score,
        "all_scores": [score for score, _ in scored_attempts],
        "selection_reason": "highest_grounding_confidence_score",
    }
    
    return best_attempt.answer, best_attempt, selection_metadata


def _compute_attempt_score(validation: Any) -> float:
    """
    Compute a composite score for an attempt based on validation metrics.
    
    Higher score indicates better answer quality.
    
    Args:
        validation: The AnswerValidationResult for the attempt
        
    Returns:
        Composite score from 0.0 to 1.0
    """
    grounded = getattr(validation, 'grounded', False)
    confidence = getattr(validation, 'confidence', 0.0)
    evidence_coverage = getattr(validation, 'evidence_coverage', 0.0)
    hallucination_detected = getattr(validation, 'hallucination_detected', False)
    contradiction_count = len(getattr(validation, 'contradictions', []))
    unsupported_claims_count = len(getattr(validation, 'unsupported_claims', []))
    
    # Base score from confidence and evidence coverage
    score = 0.5 * confidence + 0.3 * evidence_coverage
    
    # Bonus for groundedness
    if grounded:
        score += 0.15
    
    # Severe penalty for hallucination
    if hallucination_detected:
        score -= 0.4
    
    # Penalty for contradictions
    score -= min(contradiction_count * 0.1, 0.3)
    
    # Penalty for unsupported claims
    score -= min(unsupported_claims_count * 0.05, 0.2)
    
    # Clamp to [0.0, 1.0]
    score = max(0.0, min(1.0, score))
    
    return score


# ============================================================================
# RETRY EXECUTION HELPERS
# ============================================================================

def should_continue_retry(
    attempt_number: int,
    retry_decision: RetryDecision,
    current_validation: Any,
) -> bool:
    """
    Determine whether to continue retrying.
    
    This function checks if we've reached the maximum retry attempts
    or if the current attempt is good enough to stop.
    
    Args:
        attempt_number: The current attempt number (1-based)
        retry_decision: The retry decision from the first failure
        current_validation: The validation result for the current attempt
        
    Returns:
        True if should continue retrying, False otherwise
    """
    # Check max attempts
    max_attempts = retry_decision.max_retry_attempts
    if attempt_number >= max_attempts:
        return False
    
    # Check if current attempt is good enough
    grounded = getattr(current_validation, 'grounded', False)
    confidence = getattr(current_validation, 'confidence', 0.0)
    validation_action = getattr(current_validation, 'validation_action', 'accept')
    
    # Stop if answer is well-grounded
    if grounded and confidence >= 0.7 and validation_action == "accept":
        return False
    
    # Continue retrying
    return True


def build_retry_metadata(
    retry_decision: RetryDecision,
    attempt_number: int,
    previous_policy: Dict[str, Any],
    new_policy: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build metadata for a retry attempt.
    
    Args:
        retry_decision: The retry decision
        attempt_number: The attempt number
        previous_policy: The policy used in the previous attempt
        new_policy: The policy used for this attempt
        
    Returns:
        Metadata dictionary
    """
    return {
        "retried": True,
        "retry_count": attempt_number - 1,
        "retry_strategy": retry_decision.retry_strategy,
        "retry_reason": retry_decision.retry_reason,
        "retry_query": retry_decision.retry_query,
        "previous_policy": previous_policy,
        "new_policy": new_policy,
        "expanded_top_k": retry_decision.expanded_top_k,
        "use_deeper_reranking": retry_decision.use_deeper_reranking,
        "use_broader_context": retry_decision.use_broader_context,
    }
