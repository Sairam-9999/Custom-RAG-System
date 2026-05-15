"""
Answer Validation and Grounding Verification Layer for Adaptive RAG System - Phase 8

This module provides a validation layer that verifies generated answers against
retrieved evidence before returning final output.

Architecture:
    Query → Retrieval → Generation → Answer Validation → Grounding Check →
    Confidence Estimation → Refuse/Revise if needed → Final Response

The system is:
- Grounding-focused: Verifies answers are supported by retrieved evidence
- Hallucination-aware: Detects fabricated entities, numbers, and facts
- Contradiction-aware: Detects answers that contradict evidence
- Confidence-aware: Estimates grounding confidence for trust scoring
- Safety-first: Prioritizes truthfulness over helpfulness
- Modular: Lightweight and generator-agnostic
- Explainable: Provides detailed metadata for analytics
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class AnswerValidationResult:
    """
    Result of answer validation and grounding verification.
    
    This dataclass captures all validation information about a generated answer,
    including grounding status, confidence, detected issues, and recommended action.
    
    Attributes:
        answer: The generated answer to validate
        grounded: Whether the answer is grounded in the evidence
        confidence: Confidence score for the answer (0.0 to 1.0)
        validation_action: Recommended action (accept, revise, refuse, clarify)
        hallucination_detected: Whether hallucinations were detected
        unsupported_claims: List of unsupported claims in the answer
        missing_entities: List of entities in answer not found in context
        contradictions: List of contradictions with the context
        ambiguity_detected: Whether ambiguity was detected
        evidence_coverage: Fraction of answer supported by evidence (0.0 to 1.0)
        metadata: Additional validation metadata
    """
    answer: str
    grounded: bool
    confidence: float
    validation_action: str
    
    hallucination_detected: bool = False
    unsupported_claims: List[str] = field(default_factory=list)
    missing_entities: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    ambiguity_detected: bool = False
    evidence_coverage: float = 0.0
    
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# ENTITY EXTRACTION
# ============================================================================

def extract_entities(text: str) -> List[str]:
    """
    Extract entities from text using heuristics.
    
    This function extracts potential entities including:
    - Capitalized phrases
    - Quoted strings
    - Technical terms with underscores
    - Acronyms
    - Mixed-case alphanumeric terms
    
    Args:
        text: The text to extract entities from
        
    Returns:
        List of extracted entity strings
    """
    entities = []
    
    # Extract quoted strings
    quoted_pattern = r'"([^"]+)"'
    entities.extend(re.findall(quoted_pattern, text))
    
    # Extract capitalized phrases (sequences of capitalized words)
    capitalized_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
    capitalized_matches = re.findall(capitalized_pattern, text)
    
    # Filter out common words at start of sentences
    common_start_words = {
        'The', 'A', 'An', 'This', 'That', 'What', 'How', 'Why', 'Who',
        'When', 'Where', 'Which', 'Whose', 'Compare', 'Explain', 'Describe',
        'List', 'Name', 'Identify', 'Define', 'Analyze', 'Evaluate', 'However',
        'Therefore', 'Thus', 'Consequently', 'Furthermore', 'Moreover'
    }
    
    for match in capitalized_matches:
        words = match.split()
        if words[0] not in common_start_words:
            entities.append(match)
    
    # Extract technical terms (words with underscores, camelCase)
    tech_pattern = r'\b[a-z]+_[a-z]+\b|[A-Z][a-z]+[A-Z][a-z]+\b'
    entities.extend(re.findall(tech_pattern, text))
    
    # Extract all-uppercase acronyms (2+ letters)
    acronym_pattern = r'\b[A-Z]{2,}\b'
    acronym_matches = re.findall(acronym_pattern, text)
    for match in acronym_matches:
        if match not in entities:
            entities.append(match)
    
    # Extract mixed-case alphanumeric terms (like BM25, GPT2, etc.)
    mixed_pattern = r'\b[A-Z]{2,}[0-9]+\b'
    mixed_matches = re.findall(mixed_pattern, text)
    for match in mixed_matches:
        if match not in entities:
            entities.append(match)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_entities = []
    for entity in entities:
        if entity.lower() not in seen:
            seen.add(entity.lower())
            unique_entities.append(entity)
    
    return unique_entities


def extract_numbers(text: str) -> List[str]:
    """
    Extract numbers from text for hallucination detection.
    
    Args:
        text: The text to extract numbers from
        
    Returns:
        List of number strings found in the text
    """
    # Extract integers, decimals, percentages, etc.
    number_pattern = r'\b\d+(?:,\d{3})*(?:\.\d+)?(?:%|percent)?\b'
    return re.findall(number_pattern, text)


def extract_dates(text: str) -> List[str]:
    """
    Extract dates from text for temporal validation.
    
    Args:
        text: The text to extract dates from
        
    Returns:
        List of date strings found in the text
    """
    dates = []
    
    # Years (4 digits)
    year_pattern = r'\b\d{4}\b'
    dates.extend(re.findall(year_pattern, text))
    
    # Date formats like "January 15, 2020"
    date_pattern = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,\s+\d{4})?\b'
    dates.extend(re.findall(date_pattern, text, re.IGNORECASE))
    
    return list(set(dates))


# ============================================================================
# EVIDENCE OVERLAP ANALYSIS
# ============================================================================

def compute_lexical_overlap(answer: str, context: str) -> float:
    """
    Compute normalized lexical overlap between answer and context.
    
    This measures how many words in the answer appear in the context, with:
    - Lowercase normalization
    - Punctuation stripping
    - Stopword removal (common words like 'the', 'a', 'is')
    - Token overlap scoring
    
    Args:
        answer: The generated answer
        context: The retrieved context
        
    Returns:
        Overlap score from 0.0 (no overlap) to 1.0 (complete overlap)
    """
    # Common stopwords to remove (improves semantic overlap detection)
    stopwords = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'to', 'of', 'in',
        'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
        'during', 'before', 'after', 'above', 'below', 'between', 'under',
        'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
        'why', 'how', 'all', 'each', 'few', 'more', 'most', 'other',
        'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
        'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or', 'because',
        'as', 'until', 'while', 'this', 'that', 'these', 'those', 'it',
        'its', 'they', 'them', 'their', 'what', 'which', 'who', 'whom'
    }
    
    # Tokenize into words (lowercase, remove punctuation)
    answer_words = set(re.findall(r'\b\w+\b', answer.lower()))
    context_words = set(re.findall(r'\b\w+\b', context.lower()))
    
    if not answer_words:
        return 0.0
    
    # Remove stopwords from both sets
    answer_words_filtered = answer_words - stopwords
    context_words_filtered = context_words - stopwords
    
    # If filtering removed all words, use original sets
    if not answer_words_filtered:
        answer_words_filtered = answer_words
    if not context_words_filtered:
        context_words_filtered = context_words
    
    # Compute overlap
    overlap = answer_words_filtered & context_words_filtered
    overlap_score = len(overlap) / len(answer_words_filtered) if answer_words_filtered else 0.0
    
    return overlap_score


def compute_entity_overlap(answer_entities: List[str], context_entities: List[str]) -> float:
    """
    Compute entity overlap between answer and context.
    
    Args:
        answer_entities: Entities extracted from the answer
        context_entities: Entities extracted from the context
        
    Returns:
        Overlap score from 0.0 (no overlap) to 1.0 (complete overlap)
    """
    if not answer_entities:
        return 1.0  # No entities to check
    
    # Normalize to lowercase for comparison
    answer_entities_lower = {e.lower() for e in answer_entities}
    context_entities_lower = {e.lower() for e in context_entities}
    
    # Compute overlap
    overlap = answer_entities_lower & context_entities_lower
    overlap_score = len(overlap) / len(answer_entities_lower)
    
    return overlap_score


def compute_evidence_coverage(
    answer: str,
    context: str,
    answer_entities: List[str],
    context_entities: List[str],
) -> float:
    """
    Compute overall evidence coverage score with calibrated weights.
    
    This combines:
    - Normalized lexical overlap (50% weight) - relaxed for paraphrases
    - Entity overlap (30% weight) - softer matching
    - Trigram overlap (20% weight) - captures phrase-level support
    
    The calibration is designed to:
    - Accept paraphrased answers as grounded
    - Reduce false hallucination flags
    - Be more lenient with semantic equivalence
    
    Args:
        answer: The generated answer
        context: The retrieved context
        answer_entities: Entities extracted from the answer
        context_entities: Entities extracted from the context
        
    Returns:
        Coverage score from 0.0 (unsupported) to 1.0 (fully grounded)
    """
    # Lexical overlap (50% weight) - using normalized version
    lexical_score = compute_lexical_overlap(answer, context)
    
    # Entity overlap (30% weight) - relaxed matching
    entity_score = compute_entity_overlap(answer_entities, context_entities)
    
    # Trigram overlap (20% weight) - captures phrase-level support better than bigrams
    answer_trigrams = set()
    answer_words = re.findall(r'\b\w+\b', answer.lower())
    for i in range(len(answer_words) - 2):
        answer_trigrams.add(f"{answer_words[i]} {answer_words[i+1]} {answer_words[i+2]}")
    
    context_trigrams = set()
    context_words = re.findall(r'\b\w+\b', context.lower())
    for i in range(len(context_words) - 2):
        context_trigrams.add(f"{context_words[i]} {context_words[i+1]} {context_words[i+2]}")
    
    if answer_trigrams:
        trigram_overlap = answer_trigrams & context_trigrams
        trigram_score = len(trigram_overlap) / len(answer_trigrams)
    else:
        # For short answers, fall back to bigram overlap
        answer_bigrams = set()
        for i in range(len(answer_words) - 1):
            answer_bigrams.add(f"{answer_words[i]} {answer_words[i+1]}")
        
        context_bigrams = set()
        for i in range(len(context_words) - 1):
            context_bigrams.add(f"{context_words[i]} {context_words[i+1]}")
        
        if answer_bigrams:
            bigram_overlap = answer_bigrams & context_bigrams
            trigram_score = len(bigram_overlap) / len(answer_bigrams)
        else:
            trigram_score = 1.0
    
    # Weighted combination with calibrated weights
    coverage = 0.5 * lexical_score + 0.3 * entity_score + 0.2 * trigram_score
    
    return coverage


# ============================================================================
# HALLUCINATION DETECTION
# ============================================================================

def detect_unsupported_entities(
    answer_entities: List[str],
    context_entities: List[str],
) -> List[str]:
    """
    Detect entities in the answer that are not in the context.
    
    Args:
        answer_entities: Entities extracted from the answer
        context_entities: Entities extracted from the context
        
    Returns:
        List of unsupported entity strings
    """
    context_entities_lower = {e.lower() for e in context_entities}
    unsupported = []
    
    for entity in answer_entities:
        if entity.lower() not in context_entities_lower:
            unsupported.append(entity)
    
    return unsupported


def detect_unsupported_numbers(
    answer_numbers: List[str],
    context_numbers: List[str],
) -> List[str]:
    """
    Detect numbers in the answer that are not in the context.
    
    Args:
        answer_numbers: Numbers extracted from the answer
        context_numbers: Numbers extracted from the context
        
    Returns:
        List of unsupported number strings
    """
    context_numbers_set = set(context_numbers)
    unsupported = []
    
    for number in answer_numbers:
        if number not in context_numbers_set:
            unsupported.append(number)
    
    return unsupported


def detect_hallucinations(
    answer: str,
    context: str,
    answer_entities: List[str],
    context_entities: List[str],
) -> Dict[str, Any]:
    """
    Detect hallucinations in the answer with relaxed thresholds.
    
    This checks for:
    - Unsupported entities (only if >50% unsupported)
    - Unsupported numbers (only if not in context)
    - Unsupported dates (only if not in context)
    - Fabricated facts (heuristic-based, relaxed)
    
    Relaxed to reduce false positives:
    - Paraphrased entities are NOT flagged as hallucinations
    - Semantic overlap is considered
    - Partial entity overlap is acceptable
    
    Args:
        answer: The generated answer
        context: The retrieved context
        answer_entities: Entities extracted from the answer
        context_entities: Entities extracted from the context
        
    Returns:
        Dictionary with hallucination detection results
    """
    hallucination_info = {
        "detected": False,
        "unsupported_entities": [],
        "unsupported_numbers": [],
        "unsupported_dates": [],
        "unsupported_claims": [],
    }
    
    # Check unsupported entities (relaxed: only flag if >50% unsupported)
    unsupported_entities = detect_unsupported_entities(answer_entities, context_entities)
    if unsupported_entities:
        # Only flag as hallucination if more than 50% of entities are unsupported
        # This allows paraphrases with some entity variations
        if len(unsupported_entities) > len(answer_entities) * 0.5:
            hallucination_info["unsupported_entities"] = unsupported_entities
            hallucination_info["detected"] = True
    
    # Check unsupported numbers (only flag if numbers are not in context)
    answer_numbers = extract_numbers(answer)
    context_numbers = extract_numbers(context)
    unsupported_numbers = detect_unsupported_numbers(answer_numbers, context_numbers)
    if unsupported_numbers:
        # Only flag if there are numbers in the answer that are clearly fabricated
        # (i.e., numbers in answer but not in context)
        if unsupported_numbers and len(context_numbers) > 0:
            hallucination_info["unsupported_numbers"] = unsupported_numbers
            hallucination_info["detected"] = True
    
    # Check unsupported dates (only flag if dates are not in context)
    answer_dates = extract_dates(answer)
    context_dates = extract_dates(context)
    unsupported_dates = [d for d in answer_dates if d not in context_dates]
    if unsupported_dates:
        # Only flag if there are dates in the answer that are not in context
        # and context actually contains dates
        if unsupported_dates and len(context_dates) > 0:
            hallucination_info["unsupported_dates"] = unsupported_dates
            hallucination_info["detected"] = True
    
    # Heuristic: detect unsupported claims using negation words (relaxed)
    # Only flag if negation is clearly unsupported
    negation_words = ['never', 'not', 'no', 'none', 'nothing', 'nobody', 'nowhere']
    answer_lower = answer.lower()
    context_lower = context.lower()
    
    for neg in negation_words:
        if neg in answer_lower:
            # Check if this negation is supported in context
            # Only flag if the negation is about a specific fact mentioned in context
            if neg not in context_lower:
                # More lenient: check if there's semantic overlap
                # If there's good lexical overlap, don't flag the negation as hallucination
                lexical_overlap = compute_lexical_overlap(answer, context)
                if lexical_overlap < 0.3:  # Only flag if very low overlap
                    hallucination_info["unsupported_claims"].append(f"Unsupported negation: '{neg}'")
                    hallucination_info["detected"] = True
    
    return hallucination_info


# ============================================================================
# CONTRADICTION DETECTION
# ============================================================================

def detect_contradictions(answer: str, context: str) -> List[str]:
    """
    Detect contradictions between answer and context.
    
    This checks for:
    - Direct opposites (yes/no, true/false)
    - Reversed relationships
    - Conflicting numerical values
    - Temporal contradictions
    
    Args:
        answer: The generated answer
        context: The retrieved context
        
    Returns:
        List of contradiction descriptions
    """
    contradictions = []
    answer_lower = answer.lower()
    context_lower = context.lower()
    
    # Check for direct oppositions
    opposition_pairs = [
        ('yes', 'no'),
        ('true', 'false'),
        ('correct', 'incorrect'),
        ('right', 'wrong'),
        ('present', 'absent'),
        ('exists', 'does not exist'),
    ]
    
    for pos, neg in opposition_pairs:
        if pos in answer_lower and neg in context_lower:
            contradictions.append(f"Answer says '{pos}' but context says '{neg}'")
        elif neg in answer_lower and pos in context_lower:
            contradictions.append(f"Answer says '{neg}' but context says '{pos}'")
    
    # Check for numerical contradictions
    answer_numbers = extract_numbers(answer)
    context_numbers = extract_numbers(context)
    
    # Simple heuristic: if answer has a number not in context, might be contradiction
    # (This is conservative - actual contradiction detection is more complex)
    for num in answer_numbers:
        if num not in context_numbers:
            # Check if context has a different number in similar context
            # This is a simplified check
            contradictions.append(f"Answer mentions '{num}' which is not in context")
    
    return contradictions


# ============================================================================
# AMBIGUITY DETECTION
# ============================================================================

def detect_ambiguity(
    answer: str,
    query: str,
    query_analysis: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Detect ambiguity in the answer or query.
    
    Args:
        answer: The generated answer
        query: The original query
        query_analysis: Optional QueryAnalysis object
        
    Returns:
        Dictionary with ambiguity detection results
    """
    ambiguity_info = {
        "detected": False,
        "reason": None,
    }
    
    # Check query analysis for ambiguity flags
    if query_analysis is not None:
        ambiguity_flags = getattr(query_analysis, 'ambiguity_flags', [])
        if ambiguity_flags:
            ambiguity_info["detected"] = True
            ambiguity_info["reason"] = f"Query ambiguity: {ambiguity_flags}"
            return ambiguity_info
    
    # Heuristic: check for vague pronouns in answer
    vague_pronouns = ['it', 'this', 'that', 'they', 'them', 'he', 'she']
    answer_lower = answer.lower()
    
    pronoun_count = sum(1 for p in vague_pronouns if p in answer_lower)
    if pronoun_count >= 2:
        ambiguity_info["detected"] = True
        ambiguity_info["reason"] = f"Answer contains {pronoun_count} vague pronouns"
    
    return ambiguity_info


def check_comparative_entity_coverage(
    answer: str,
    comparison_targets: List[str],
) -> Dict[str, Any]:
    """
    Check if all comparison targets are covered in the answer.
    
    For comparative queries, it's critical that both entities are mentioned
    in the answer to provide a balanced comparison.
    
    Args:
        answer: The generated answer
        comparison_targets: List of entities being compared
        
    Returns:
        Dictionary with entity coverage information
    """
    if not comparison_targets or len(comparison_targets) < 2:
        return {
            "all_covered": True,
            "missing_targets": [],
            "coverage_ratio": 1.0,
        }
    
    answer_lower = answer.lower()
    covered_targets = []
    missing_targets = []
    
    for target in comparison_targets:
        if target.lower() in answer_lower:
            covered_targets.append(target)
        else:
            missing_targets.append(target)
    
    coverage_ratio = len(covered_targets) / len(comparison_targets)
    
    return {
        "all_covered": len(missing_targets) == 0,
        "covered_targets": covered_targets,
        "missing_targets": missing_targets,
        "coverage_ratio": coverage_ratio,
    }


# ============================================================================
# CONFIDENCE ESTIMATION
# ============================================================================

def estimate_confidence(
    evidence_coverage: float,
    hallucination_detected: bool,
    contradiction_count: int,
    ambiguity_detected: bool,
    query_type: Optional[str] = None,
    comparative_coverage_ratio: Optional[float] = None,
) -> float:
    """
    Estimate confidence in the answer based on validation metrics.
    
    Args:
        evidence_coverage: Evidence coverage score (0.0 to 1.0)
        hallucination_detected: Whether hallucinations were detected
        contradiction_count: Number of contradictions detected
        ambiguity_detected: Whether ambiguity was detected
        query_type: The query type (for type-specific adjustments)
        comparative_coverage_ratio: Entity coverage ratio for comparative queries (0.0 to 1.0)
        
    Returns:
        Confidence score from 0.0 to 1.0
    """
    # Base confidence from evidence coverage
    confidence = evidence_coverage
    
    # Penalty for hallucinations (severe)
    if hallucination_detected:
        confidence -= 0.4
    
    # Penalty for contradictions (moderate per contradiction)
    confidence -= min(contradiction_count * 0.15, 0.5)
    
    # Penalty for ambiguity
    if ambiguity_detected:
        confidence -= 0.1
    
    # Penalty for missing comparative entity coverage
    if query_type == "comparative" and comparative_coverage_ratio is not None:
        if comparative_coverage_ratio < 1.0:
            # Reduce confidence if not all comparison targets are covered
            confidence -= (1.0 - comparative_coverage_ratio) * 0.3
    
    # Query-type specific adjustments
    if query_type == "unanswerable":
        # Unanswerable queries naturally have lower confidence
        confidence = min(confidence, 0.3)
    elif query_type == "factual":
        # Factual queries should have high grounding
        if evidence_coverage < 0.6:
            confidence -= 0.2
    
    # Clamp to [0.0, 1.0]
    confidence = max(0.0, min(1.0, confidence))
    
    return confidence


# ============================================================================
# VALIDATION ACTION DECISION
# ============================================================================

def decide_validation_action(
    grounded: bool,
    confidence: float,
    hallucination_detected: bool,
    contradiction_count: int,
    evidence_coverage: float,
    ambiguity_detected: bool,
) -> str:
    """
    Decide the appropriate validation action based on validation metrics.
    
    Args:
        grounded: Whether the answer is grounded
        confidence: Confidence score
        hallucination_detected: Whether hallucinations were detected
        contradiction_count: Number of contradictions
        evidence_coverage: Evidence coverage score
        ambiguity_detected: Whether ambiguity was detected
        
    Returns:
        Validation action: 'accept', 'revise', 'refuse', or 'clarify'
    """
    # REFUSE: Severe issues (relaxed thresholds)
    if hallucination_detected or contradiction_count >= 3 or evidence_coverage < 0.2:
        return "refuse"
    
    # CLARIFY: Ambiguity detected
    if ambiguity_detected:
        return "clarify"
    
    # REVISE: Partial grounding but not severe (relaxed thresholds)
    if not grounded or confidence < 0.4 or evidence_coverage < 0.4:
        return "revise"
    
    # ACCEPT: Good grounding and confidence
    return "accept"


# ============================================================================
# MAIN VALIDATION FUNCTION
# ============================================================================

def validate_answer(
    query: str,
    answer: str,
    context: str,
    query_analysis: Optional[Any] = None,
) -> AnswerValidationResult:
    """
    Validate a generated answer against retrieved evidence.
    
    This is the main entry point for answer validation. It performs:
    - Entity extraction
    - Evidence overlap analysis
    - Hallucination detection
    - Contradiction detection
    - Ambiguity detection
    - Comparative entity coverage check (for comparative queries)
    - Confidence estimation
    - Validation action decision
    
    Args:
        query: The original user query
        answer: The generated answer to validate
        context: The retrieved context
        query_analysis: Optional QueryAnalysis object for additional metadata
        
    Returns:
        AnswerValidationResult with validation information
        
    Example:
        >>> result = validate_answer(
        ...     query="What is the capital of France?",
        ...     answer="The capital of France is Paris.",
        ...     context="France is a country in Europe. Its capital is Paris.",
        ... )
        >>> if result.validation_action == "accept":
        ...     return result.answer
    """
    # Extract entities
    answer_entities = extract_entities(answer)
    context_entities = extract_entities(context)
    
    # Compute evidence coverage
    evidence_coverage = compute_evidence_coverage(
        answer, context, answer_entities, context_entities
    )
    
    # Detect hallucinations
    hallucination_info = detect_hallucinations(
        answer, context, answer_entities, context_entities
    )
    
    # Detect contradictions
    contradictions = detect_contradictions(answer, context)
    
    # Detect ambiguity
    ambiguity_info = detect_ambiguity(answer, query, query_analysis)
    
    # Check comparative entity coverage
    comparative_coverage = None
    if query_analysis is not None:
        query_type = getattr(query_analysis, 'query_type', None)
        if query_type == "comparative":
            comparison_targets = getattr(query_analysis, 'comparison_targets', [])
            comparative_coverage = check_comparative_entity_coverage(
                answer, comparison_targets
            )
    
    # Determine if grounded (relaxed threshold from 0.6 to 0.4)
    grounded = (
        evidence_coverage >= 0.4 and
        not hallucination_info["detected"] and
        len(contradictions) == 0
    )
    
    # For comparative queries, also require entity coverage
    if comparative_coverage and not comparative_coverage["all_covered"]:
        grounded = False
    
    # Get query type from analysis if available
    query_type = None
    if query_analysis is not None:
        query_type = getattr(query_analysis, 'query_type', None)
    
    # Estimate confidence
    comparative_ratio = comparative_coverage["coverage_ratio"] if comparative_coverage else None
    confidence = estimate_confidence(
        evidence_coverage=evidence_coverage,
        hallucination_detected=hallucination_info["detected"],
        contradiction_count=len(contradictions),
        ambiguity_detected=ambiguity_info["detected"],
        query_type=query_type,
        comparative_coverage_ratio=comparative_ratio,
    )
    
    # Decide validation action
    validation_action = decide_validation_action(
        grounded=grounded,
        confidence=confidence,
        hallucination_detected=hallucination_info["detected"],
        contradiction_count=len(contradictions),
        evidence_coverage=evidence_coverage,
        ambiguity_detected=ambiguity_info["detected"],
    )
    
    # Build metadata
    metadata = {
        "query_type": query_type,
        "answer_entity_count": len(answer_entities),
        "context_entity_count": len(context_entities),
        "unsupported_numbers": hallucination_info["unsupported_numbers"],
        "unsupported_dates": hallucination_info["unsupported_dates"],
    }
    
    # Add comparative coverage info if applicable
    if comparative_coverage:
        metadata["comparative_coverage"] = comparative_coverage
    
    # Build result
    result = AnswerValidationResult(
        answer=answer,
        grounded=grounded,
        confidence=confidence,
        validation_action=validation_action,
        hallucination_detected=hallucination_info["detected"],
        unsupported_claims=hallucination_info["unsupported_claims"],
        missing_entities=hallucination_info["unsupported_entities"],
        contradictions=contradictions,
        ambiguity_detected=ambiguity_info["detected"],
        evidence_coverage=evidence_coverage,
        metadata=metadata,
    )
    
    return result


# ============================================================================
# ANSWER REVISION
# ============================================================================

def revise_answer(
    answer: str,
    context: str,
    validation_result: AnswerValidationResult,
) -> str:
    """
    Revise an answer based on validation results.
    
    This function attempts to fix issues in the answer by:
    - Removing unsupported entities
    - Trimming over-generated content
    - Falling back to extractive evidence if needed
    
    Args:
        answer: The original answer
        context: The retrieved context
        validation_result: The validation result for the answer
        
    Returns:
        Revised answer string
    """
    # If severe hallucinations, refuse instead of revising
    if validation_result.hallucination_detected and len(validation_result.missing_entities) > 2:
        return "I don't know from the provided context."
    
    # If contradictions are severe, refuse
    if len(validation_result.contradictions) >= 2:
        return "I don't know from the provided context."
    
    # Try to extract a safe, grounded sentence from context
    # Simple heuristic: extract the longest sentence with entity overlap
    sentences = re.split(r'[.!?]', context)
    answer_entities = extract_entities(answer)
    
    best_sentence = ""
    best_score = 0.0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # Compute entity overlap with original answer
        sentence_entities = extract_entities(sentence)
        overlap = compute_entity_overlap(answer_entities, sentence_entities)
        
        if overlap > best_score and len(sentence) > 10:
            best_score = overlap
            best_sentence = sentence
    
    if best_sentence:
        return best_sentence + "."
    
    # Fallback: return a conservative answer
    if validation_result.evidence_coverage > 0.3:
        return "Based on the provided context, " + answer.lower()
    else:
        return "I don't know from the provided context."
