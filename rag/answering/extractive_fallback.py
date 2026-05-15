"""
Query-Aware Extractive Fallback System - Phase 11.6

This module provides an intelligent fallback answer selector that understands
query intent and selects the most semantically appropriate grounded evidence.

Architecture:
    Generation Failure → Query-Aware Fallback Planner → Intent-Aware Sentence Selection
    → Grounded Extractive Synthesis → Final Safe Answer

The system is:
- Query-type aware (factual, temporal, comparative, multi-hop, analytical, procedural, unanswerable)
- Evidence-aware (keyword overlap, entity matching, temporal relationships)
- Grounded (avoids hallucination, uses only provided context)
- Lightweight (deterministic heuristics, no LLM dependencies)
- Explainable (transparent scoring and selection logic)
- Safe (refuses unanswerable queries gracefully)
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
from enum import Enum


class QueryType(Enum):
    """Query type enumeration for strategy selection."""
    FACTUAL = "factual"
    TEMPORAL = "temporal"
    COMPARATIVE = "comparative"
    MULTI_HOP = "multi_hop"
    ANALYTICAL = "analytical"
    PROCEDURAL = "procedural"
    UNANSWERABLE = "unanswerable"


@dataclass
class FallbackResult:
    """
    Result of the extractive fallback process.
    
    Attributes:
        answer: The extracted/synthesized fallback answer
        confidence: Confidence score (0.0 to 1.0)
        strategy: The fallback strategy used
        selected_sentences: List of sentences selected for the answer
        sentence_scores: Scores for each sentence
        query_type: The detected query type
        metadata: Additional debugging metadata
    """
    answer: str
    confidence: float
    strategy: str
    selected_sentences: List[str] = field(default_factory=list)
    sentence_scores: List[Tuple[str, float]] = field(default_factory=list)
    query_type: str = "factual"
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# SENTENCE EXTRACTION UTILITIES
# ============================================================================

def split_sentences(context: str) -> List[str]:
    """
    Split context into sentences using robust punctuation-based splitting.
    
    Args:
        context: Text context to split
        
    Returns:
        List of sentence strings
    """
    # Split on sentence boundaries (. ! ?) followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', context)
    return [s.strip() for s in sentences if s.strip()]


def extract_noun_phrases(sentence: str) -> List[str]:
    """
    Extract simple noun phrases from a sentence.
    
    Uses heuristic patterns to identify potential noun phrases.
    
    Args:
        sentence: Sentence to analyze
        
    Returns:
        List of noun phrase strings
    """
    # Pattern: optional determiner + adjectives + noun
    pattern = r'\b(?:a|an|the)?\s*(?:\w+\s+)*?\w+(?:\s+\w+)?\b'
    matches = re.findall(pattern, sentence, re.IGNORECASE)
    return [m.strip() for m in matches if len(m.split()) >= 2]


# ============================================================================
# SCORING COMPONENTS
# ============================================================================

def calculate_keyword_overlap(
    sentence: str,
    keywords: List[str],
    entities: List[str],
) -> float:
    """
    Calculate keyword and entity overlap score.
    
    Args:
        sentence: Sentence to score
        keywords: Query keywords
        entities: Query entities
        
    Returns:
        Overlap score (0.0 to 1.0)
    """
    sentence_lower = sentence.lower()
    sentence_words = set(re.findall(r'\b\w+\b', sentence_lower))
    
    # Keyword overlap
    keyword_set = set(k.lower() for k in keywords)
    keyword_matches = len(keyword_set.intersection(sentence_words))
    keyword_score = keyword_matches / max(len(keyword_set), 1)
    
    # Entity overlap (higher weight)
    entity_matches = 0
    for entity in entities:
        if entity.lower() in sentence_lower:
            entity_matches += 1
    entity_score = entity_matches / max(len(entities), 1)
    
    # Combine: entities weighted higher (2x)
    total_score = (keyword_score + 2 * entity_score) / 3
    
    return min(total_score, 1.0)


def calculate_temporal_boost(
    sentence: str,
    temporal_markers: List[str],
    query_type: QueryType,
) -> float:
    """
    Calculate temporal relevance boost for sentences.
    
    Args:
        sentence: Sentence to score
        temporal_markers: Temporal markers from query
        query_type: Detected query type
        
    Returns:
        Temporal boost score (0.0 to 1.0)
    """
    if query_type != QueryType.TEMPORAL and not temporal_markers:
        return 0.0
    
    sentence_lower = sentence.lower()
    
    # Temporal transition words in context
    temporal_words = [
        'after', 'when', 'then', 'before', 'yet when', 'subsequently',
        'later', 'finally', 'eventually', 'following', 'resulted'
    ]
    
    temporal_in_sentence = sum(1 for word in temporal_words if word in sentence_lower)
    temporal_score = min(temporal_in_sentence / 2, 1.0)
    
    return temporal_score


def calculate_comparative_balance(
    sentence: str,
    comparison_targets: List[str],
) -> float:
    """
    Calculate comparative balance - ensure both comparison targets are represented.
    
    Args:
        sentence: Sentence to score
        comparison_targets: Entities being compared
        
    Returns:
        Balance score (0.0 to 1.0)
    """
    if not comparison_targets or len(comparison_targets) < 2:
        return 0.0
    
    sentence_lower = sentence.lower()
    targets_present = sum(1 for target in comparison_targets if target.lower() in sentence_lower)
    
    # Prefer sentences that mention both targets
    if targets_present >= 2:
        return 1.0
    elif targets_present == 1:
        return 0.5
    else:
        return 0.0


def calculate_positional_relevance(
    sentence: str,
    sentence_index: int,
    total_sentences: int,
    query_type: QueryType,
    question: str,
) -> float:
    """
    Calculate positional relevance based on query structure.
    
    For temporal queries like "what happened after X", boost later sentences
    that represent consequences.
    
    Args:
        sentence: Sentence to score
        sentence_index: Index of sentence in context (0-based)
        total_sentences: Total number of sentences
        query_type: Detected query type
        question: Original question
        
    Returns:
        Positional relevance score (0.0 to 1.0)
    """
    question_lower = question.lower()
    
    # For "after" queries, prefer later sentences (consequences)
    if query_type == QueryType.TEMPORAL and 'after' in question_lower:
        # Normalize position: later sentences get higher scores
        position_score = (sentence_index + 1) / max(total_sentences, 1)
        return position_score
    
    # For "before" queries, prefer earlier sentences
    if query_type == QueryType.TEMPORAL and 'before' in question_lower:
        position_score = 1.0 - (sentence_index / max(total_sentences, 1))
        return position_score
    
    # Default: slight preference for earlier sentences (background info)
    return 1.0 - (sentence_index * 0.1 / max(total_sentences, 1))


def calculate_answerability_signals(sentence: str) -> float:
    """
    Detect answerability signals in sentences.
    
    Boost sentences that contain causal, explanatory, or event-result structures.
    
    Args:
        sentence: Sentence to score
        
    Returns:
        Answerability signal score (0.0 to 1.0)
    """
    sentence_lower = sentence.lower()
    
    # Causal/explanatory patterns
    causal_patterns = [
        'because', 'since', 'due to', 'as a result', 'consequently',
        'therefore', 'thus', 'led to', 'caused', 'resulted in'
    ]
    
    # Event-result patterns
    event_result_patterns = [
        'found', 'discovered', 'realized', 'warned', 'remembered',
        'occurred', 'happened', 'took place', 'emerged'
    ]
    
    # Definition/fact patterns
    fact_patterns = [
        'is', 'are', 'was', 'were', 'refers to', 'means', 'defined as'
    ]
    
    causal_count = sum(1 for pattern in causal_patterns if pattern in sentence_lower)
    event_count = sum(1 for pattern in event_result_patterns if pattern in sentence_lower)
    fact_count = sum(1 for pattern in fact_patterns if pattern in sentence_lower)
    
    # Calculate combined signal score
    signal_score = min((causal_count * 0.4 + event_count * 0.4 + fact_count * 0.2), 1.0)
    
    return signal_score


def calculate_warning_consequence_boost(sentence: str, question: str) -> float:
    """
    Calculate boost for warning and consequence sentences in multi-hop queries.
    
    For questions asking about warnings, consequences, or "what happened",
    boost sentences containing warning/consequence indicators.
    
    Args:
        sentence: Sentence to score
        question: Original question
        
    Returns:
        Warning/consequence boost score (0.0 to 0.5)
    """
    question_lower = question.lower()
    sentence_lower = sentence.lower()
    
    # Check if question asks about warnings/consequences
    warning_question_patterns = [
        'warn', 'warning', 'what did they warn', 'what happened', 'why',
        'consequence', 'result', 'outcome', 'effect'
    ]
    
    asks_about_warning = any(pattern in question_lower for pattern in warning_question_patterns)
    
    if not asks_about_warning:
        return 0.0
    
    # Warning/consequence indicators in sentence
    warning_indicators = [
        'warned', 'warning', 'disappear', 'because', 'therefore', 'resulted',
        'caused', 'led to', 'consequently', 'thus', 'as a result'
    ]
    
    indicator_count = sum(1 for indicator in warning_indicators if indicator in sentence_lower)
    boost = min(indicator_count * 0.25, 0.5)
    
    return boost


def calculate_sentence_similarity(sentence1: str, sentence2: str) -> float:
    """
    Calculate semantic similarity between two sentences using Jaccard similarity.
    
    Args:
        sentence1: First sentence
        sentence2: Second sentence
        
    Returns:
        Similarity score (0.0 to 1.0)
    """
    words1 = set(re.findall(r'\b\w+\b', sentence1.lower()))
    words2 = set(re.findall(r'\b\w+\b', sentence2.lower()))
    
    if not words1 and not words2:
        return 1.0
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 0.0


def deduplicate_sentences(sentences: List[str], threshold: float = 0.75) -> List[str]:
    """
    Remove semantically similar sentences from a list.
    
    Args:
        sentences: List of sentences to deduplicate
        threshold: Similarity threshold above which sentences are considered duplicates
        
    Returns:
        Deduplicated list of sentences
    """
    if not sentences:
        return []
    
    deduplicated = [sentences[0]]
    
    for sentence in sentences[1:]:
        is_duplicate = False
        for existing in deduplicated:
            similarity = calculate_sentence_similarity(sentence, existing)
            if similarity >= threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            deduplicated.append(sentence)
    
    return deduplicated


def merge_structurally_identical_sentences(sentences: List[str]) -> List[str]:
    """
    Merge structurally identical sentences with different entities.
    
    Example: "Lyra remembered..." and "Selis remembered..." -> "Lyra, Selis, and Kaedrin remembered..."
    
    Args:
        sentences: List of sentences to merge
        
    Returns:
        List of merged sentences
    """
    if len(sentences) < 2:
        return sentences
    
    # Group sentences by structure (same pattern, different entities)
    structure_groups = {}
    
    for sentence in sentences:
        # Extract structure by replacing named entities with placeholder
        # Simple heuristic: replace capitalized words with placeholder
        structure = re.sub(r'\b[A-Z][a-z]+\b', 'ENTITY', sentence)
        
        if structure not in structure_groups:
            structure_groups[structure] = []
        structure_groups[structure].append(sentence)
    
    merged = []
    for structure, group in structure_groups.items():
        if len(group) == 1:
            merged.append(group[0])
        else:
            # Extract entities from each sentence
            entities_per_sentence = []
            for sentence in group:
                entities = re.findall(r'\b[A-Z][a-z]+\b', sentence)
                entities_per_sentence.append(entities)
            
            # Check if structure is truly identical (same pattern, just different entities)
            # Only merge if all sentences have the same structure
            if len(structure_groups) == 1 and len(entities_per_sentence) > 1:
                # Merge entities
                all_entities = [e for entities in entities_per_sentence for e in entities]
                unique_entities = list(dict.fromkeys(all_entities))  # Preserve order, remove duplicates
                
                if len(unique_entities) > 2:
                    entity_str = ", ".join(unique_entities[:-1]) + ", and " + unique_entities[-1]
                elif len(unique_entities) == 2:
                    entity_str = " and ".join(unique_entities)
                else:
                    entity_str = unique_entities[0]
                
                # Replace first ENTITY with merged entities
                merged_sentence = re.sub(r'\b[A-Z][a-z]+\b', entity_str, group[0], count=1)
                merged.append(merged_sentence)
            else:
                merged.extend(group)
    
    return merged


# ============================================================================
# QUERY-TYPE SPECIFIC STRATEGIES
# ============================================================================

def select_factual_sentences(
    sentences: List[str],
    scores: List[float],
    keywords: List[str],
    entities: List[str],
    max_sentences: int = 2,
) -> Tuple[List[str], List[Tuple[str, float]]]:
    """
    Select sentences for factual queries.
    
    Strategy: Select most direct answer sentence with exact entity overlap.
    Prioritize definition/fact statements.
    
    Args:
        sentences: List of candidate sentences
        scores: Pre-computed scores for each sentence
        keywords: Query keywords
        entities: Query entities
        max_sentences: Maximum sentences to select
        
    Returns:
        Tuple of (selected_sentences, scored_sentences)
    """
    # Sort by score
    scored_sentences = list(zip(sentences, scores))
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    
    # Select top sentences
    selected = scored_sentences[:max_sentences]
    selected_sentences = [s for s, score in selected]
    
    return selected_sentences, scored_sentences


def select_temporal_sentences(
    sentences: List[str],
    scores: List[float],
    question: str,
    max_sentences: int = 2,
) -> Tuple[List[str], List[Tuple[str, float]]]:
    """
    Select sentences for temporal queries.
    
    Strategy: Prioritize AFTER/BEFORE sentences, preserve chronology,
    select event-result relationships.
    
    Args:
        sentences: List of candidate sentences
        scores: Pre-computed scores for each sentence
        question: Original question
        max_sentences: Maximum sentences to select
        
    Returns:
        Tuple of (selected_sentences, scored_sentences)
    """
    question_lower = question.lower()
    
    # Identify temporal preference
    prefer_after = 'after' in question_lower
    prefer_before = 'before' in question_lower
    
    # Boost scores based on temporal markers
    boosted_scores = []
    for i, (sentence, base_score) in enumerate(zip(sentences, scores)):
        sentence_lower = sentence.lower()
        
        # Temporal transition words
        temporal_boost = 0.0
        if prefer_after:
            if any(word in sentence_lower for word in ['after', 'when', 'then', 'subsequently']):
                temporal_boost = 0.3
        elif prefer_before:
            if any(word in sentence_lower for word in ['before', 'prior', 'earlier']):
                temporal_boost = 0.3
        
        boosted_scores.append(base_score + temporal_boost)
    
    # Sort by boosted score
    scored_sentences = list(zip(sentences, boosted_scores))
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    
    # Select top sentences
    selected = scored_sentences[:max_sentences]
    selected_sentences = [s for s, score in selected]
    
    return selected_sentences, scored_sentences


def select_comparative_sentences(
    sentences: List[str],
    scores: List[float],
    comparison_targets: List[str],
    max_sentences: int = 3,
) -> Tuple[List[str], List[Tuple[str, float]]]:
    """
    Select sentences for comparative queries.
    
    Strategy: Ensure both entities are represented, avoid one-sided extraction,
    merge balanced evidence. Now includes entity preservation enforcement.
    
    Args:
        sentences: List of candidate sentences
        scores: Pre-computed scores for each sentence
        comparison_targets: Entities being compared
        max_sentences: Maximum sentences to select
        
    Returns:
        Tuple of (selected_sentences, scored_sentences)
    """
    if not comparison_targets or len(comparison_targets) < 2:
        # Fall back to factual selection
        return select_factual_sentences(sentences, scores, [], [], max_sentences)
    
    # Calculate balance scores
    balanced_scores = []
    for sentence, base_score in zip(sentences, scores):
        balance = calculate_comparative_balance(sentence, comparison_targets)
        # Boost balanced sentences
        balanced_score = base_score + (balance * 0.5)
        balanced_scores.append(balanced_score)
    
    # Sort by balanced score
    scored_sentences = list(zip(sentences, balanced_scores))
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    
    # Group sentences by entity presence
    entity_groups = {target: [] for target in comparison_targets}
    balanced_sentences = []  # Sentences with both entities
    
    for sentence, score in scored_sentences:
        sentence_lower = sentence.lower()
        targets_mentioned = [
            target for target in comparison_targets if target.lower() in sentence_lower
        ]
        
        if len(targets_mentioned) >= 2:
            balanced_sentences.append((sentence, score))
        else:
            for target in targets_mentioned:
                entity_groups[target].append((sentence, score))
    
    # Select with entity balancing
    selected_sentences = []
    
    # First, prioritize balanced sentences (both entities)
    for sentence, score in balanced_sentences[:max_sentences]:
        selected_sentences.append(sentence)
        if len(selected_sentences) >= max_sentences:
            break
    
    # If we still need sentences, ensure both entities are represented
    if len(selected_sentences) < max_sentences:
        # Check which entities are missing
        selected_text_lower = " ".join(selected_sentences).lower()
        missing_entities = [
            target for target in comparison_targets if target.lower() not in selected_text_lower
        ]
        
        # Add best sentence for each missing entity
        for missing_entity in missing_entities:
            if len(selected_sentences) >= max_sentences:
                break
            if entity_groups[missing_entity]:
                best_sentence = entity_groups[missing_entity][0][0]
                if best_sentence not in selected_sentences:
                    selected_sentences.append(best_sentence)
    
    # Fill remaining slots with best remaining sentences
    remaining_sentences = [s for s in scored_sentences if s[0] not in selected_sentences]
    while len(selected_sentences) < max_sentences and remaining_sentences:
        next_sentence = remaining_sentences.pop(0)[0]
        if next_sentence not in selected_sentences:
            selected_sentences.append(next_sentence)
    
    return selected_sentences, scored_sentences


def select_multi_hop_sentences(
    sentences: List[str],
    scores: List[float],
    entities: List[str],
    question: str = "",
    max_sentences: int = 3,
) -> Tuple[List[str], List[Tuple[str, float]]]:
    """
    Select sentences for multi-hop queries.
    
    Strategy: Combine multiple supporting sentences, connect entity + event,
    synthesize carefully without hallucination. Now includes:
    - Warning/consequence sentence boosting
    - Complementary evidence prioritization
    - Semantic deduplication
    - Lightweight entity merging
    
    Args:
        sentences: List of candidate sentences
        scores: Pre-computed scores for each sentence
        entities: Query entities
        question: Original question (for warning/consequence detection)
        max_sentences: Maximum sentences to select
        
    Returns:
        Tuple of (selected_sentences, scored_sentences)
    """
    if not entities:
        return select_factual_sentences(sentences, scores, [], [], max_sentences)
    
    # Boost warning/consequence sentences
    boosted_scores = []
    for sentence, base_score in zip(sentences, scores):
        warning_boost = calculate_warning_consequence_boost(sentence, question)
        boosted_scores.append(base_score + warning_boost)
    
    # Update scores
    scores = boosted_scores
    
    # Group sentences by entity presence and type
    entity_sentences = []  # Sentences with entities
    consequence_sentences = []  # Sentences with warnings/consequences
    
    for sentence, score in zip(sentences, scores):
        sentence_lower = sentence.lower()
        
        # Check if sentence contains entities
        has_entity = any(entity.lower() in sentence_lower for entity in entities)
        
        # Check if sentence contains consequence indicators
        warning_indicators = ['warned', 'warning', 'disappear', 'because', 'therefore', 
                              'resulted', 'caused', 'led to', 'consequently', 'thus']
        has_consequence = any(indicator in sentence_lower for indicator in warning_indicators)
        
        if has_entity:
            entity_sentences.append((sentence, score))
        if has_consequence:
            consequence_sentences.append((sentence, score))
    
    # Sort by score
    entity_sentences.sort(key=lambda x: x[1], reverse=True)
    consequence_sentences.sort(key=lambda x: x[1], reverse=True)
    
    # Select complementary evidence: entity sentence + consequence sentence
    selected_sentences = []
    
    # Add best entity sentence
    if entity_sentences:
        selected_sentences.append(entity_sentences[0][0])
    
    # Add best consequence sentence (if different from entity sentence)
    if consequence_sentences:
        best_consequence = consequence_sentences[0][0]
        if best_consequence not in selected_sentences:
            selected_sentences.append(best_consequence)
    
    # Add more entity sentences if needed
    remaining_entity_sentences = [s for s in entity_sentences[1:] if s[0] not in selected_sentences]
    
    for sentence, score in remaining_entity_sentences:
        if len(selected_sentences) >= max_sentences:
            break
        selected_sentences.append(sentence)
    
    # Deduplicate semantically similar sentences
    selected_sentences = deduplicate_sentences(selected_sentences, threshold=0.75)
    
    # Merge structurally identical sentences
    selected_sentences = merge_structurally_identical_sentences(selected_sentences)
    
    # Rebuild scored_sentences for output
    scored_sentences = list(zip(sentences, scores))
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    
    return selected_sentences, scored_sentences


def select_analytical_sentences(
    sentences: List[str],
    scores: List[float],
    keywords: List[str],
    max_sentences: int = 2,
) -> Tuple[List[str], List[Tuple[str, float]]]:
    """
    Select sentences for analytical queries.
    
    Strategy: Avoid naive extraction, use cautious grounded interpretation.
    If insufficient evidence, return cautious template.
    
    Args:
        sentences: List of candidate sentences
        scores: Pre-computed scores for each sentence
        keywords: Query keywords
        max_sentences: Maximum sentences to select
        
    Returns:
        Tuple of (selected_sentences, scored_sentences)
    """
    # Analytical queries need thematic evidence
    # Look for sentences with interpretive keywords
    interpretive_keywords = [
        'symbolize', 'meaning', 'interpret', 'theme', 'represent',
        'significance', 'metaphor', 'allegory', 'implication'
    ]
    
    # Boost sentences with interpretive content
    boosted_scores = []
    for sentence, base_score in zip(sentences, scores):
        sentence_lower = sentence.lower()
        interpretive_boost = 0.0
        for keyword in interpretive_keywords:
            if keyword in sentence_lower:
                interpretive_boost = 0.3
                break
        boosted_scores.append(base_score + interpretive_boost)
    
    # Sort by boosted score
    scored_sentences = list(zip(sentences, boosted_scores))
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    
    # Select top sentences
    selected = scored_sentences[:max_sentences]
    selected_sentences = [s for s, score in selected]
    
    return selected_sentences, scored_sentences


def select_procedural_sentences(
    sentences: List[str],
    scores: List[float],
    max_sentences: int = 3,
) -> Tuple[List[str], List[Tuple[str, float]]]:
    """
    Select sentences for procedural queries.
    
    Strategy: Preserve sequential order, prefer instructional sentences,
    prefer process-oriented language.
    
    Args:
        sentences: List of candidate sentences
        scores: Pre-computed scores for each sentence
        max_sentences: Maximum sentences to select
        
    Returns:
        Tuple of (selected_sentences, scored_sentences)
    """
    # Look for procedural markers
    procedural_markers = [
        'step', 'first', 'then', 'next', 'finally', 'after', 'before',
        'process', 'method', 'procedure', 'workflow', 'implement',
        'configure', 'setup', 'execute', 'perform'
    ]
    
    # Boost sentences with procedural content
    boosted_scores = []
    for sentence, base_score in zip(sentences, scores):
        sentence_lower = sentence.lower()
        procedural_boost = 0.0
        for marker in procedural_markers:
            if marker in sentence_lower:
                procedural_boost = 0.3
                break
        boosted_scores.append(base_score + procedural_boost)
    
    # Sort by boosted score AND preserve original order for sequence
    indexed_sentences = list(enumerate(zip(sentences, boosted_scores)))
    
    # Sort by score first
    indexed_sentences.sort(key=lambda x: x[1][1], reverse=True)
    
    # Select top sentences
    selected_indexed = indexed_sentences[:max_sentences]
    
    # Re-sort by original index to preserve sequence
    selected_indexed.sort(key=lambda x: x[0])
    
    selected_sentences = [s for i, (s, score) in selected_indexed]
    scored_sentences = list(zip(sentences, boosted_scores))
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    
    return selected_sentences, scored_sentences


def select_unanswerable_sentences(
    sentences: List[str],
    scores: List[float],
    keywords: List[str],
    entities: List[str],
) -> Tuple[List[str], List[Tuple[str, float]]]:
    """
    Handle unanswerable queries.
    
    Strategy: Refuse safely, do not attempt extraction.
    
    Args:
        sentences: List of candidate sentences
        scores: Pre-computed scores for each sentence
        keywords: Query keywords
        entities: Query entities
        
    Returns:
        Tuple of (selected_sentences, scored_sentences) - both empty for unanswerable
    """
    return [], []


# ============================================================================
# LIGHTWEIGHT SYNTHESIS
# ============================================================================

def synthesize_answer(
    selected_sentences: List[str],
    query_type: QueryType,
    question: str,
) -> str:
    """
    Synthesize a concise answer from selected sentences.
    
    Strategy: Combine directly supported sentences, avoid hallucination,
    preserve semantic connections.
    
    Args:
        selected_sentences: Sentences selected for the answer
        query_type: Detected query type
        question: Original question
        
    Returns:
        Synthesized answer string
    """
    if not selected_sentences:
        return "I don't know from the provided context."
    
    # For single sentence, return as-is
    if len(selected_sentences) == 1:
        return selected_sentences[0].strip()
    
    # For multiple sentences, combine with careful synthesis
    # Check if sentences are semantically connected
    combined = " ".join(selected_sentences)
    
    # For multi-hop, ensure we're connecting related information
    if query_type == QueryType.MULTI_HOP:
        # Join sentences that flow together
        return combined
    
    # For comparative, ensure balance
    if query_type == QueryType.COMPARATIVE:
        return combined
    
    # Default: join sentences
    return combined


def generate_analytical_fallback(
    selected_sentences: List[str],
    keywords: List[str],
) -> str:
    """
    Generate cautious analytical fallback when evidence is weak.
    
    Uses interpretive templates grounded in available evidence.
    
    Args:
        selected_sentences: Sentences with some evidence
        keywords: Query keywords
        
    Returns:
        Cautious interpretive answer
    """
    if not selected_sentences:
        return "The context does not provide a definitive interpretation."
    
    # Look for thematic patterns in selected sentences
    combined_text = " ".join(selected_sentences).lower()
    
    # Common themes in narrative text
    theme_mappings = {
        'collapse': ['collapse', 'fall', 'ruin', 'destroy', 'broken'],
        'fear': ['fear', 'terror', 'dread', 'afraid', 'scared'],
        'instability': ['instability', 'unstable', 'chaos', 'disorder'],
        'hope': ['hope', 'hopeful', 'optimism', 'believe'],
        'sacrifice': ['sacrifice', 'gave up', 'offering'],
    }
    
    detected_themes = []
    for theme, indicators in theme_mappings.items():
        if any(indicator in combined_text for indicator in indicators):
            detected_themes.append(theme)
    
    if detected_themes:
        themes_str = ", ".join(detected_themes)
        return f"The context suggests themes of {themes_str}, but does not provide a definitive interpretation."
    
    # Default cautious response
    return "The context suggests relevant themes, but does not provide a definitive interpretation."


# ============================================================================
# MAIN FALLBACK API
# ============================================================================

def extractive_fallback(
    question: str,
    context: str,
    query_analysis: Optional[Any] = None,
    max_sentences: int = 2,
    enable_debug: bool = False,
) -> FallbackResult:
    """
    Query-aware extractive fallback for generation failures.
    
    Selects the most semantically appropriate grounded evidence based on
    query intent and query structure.
    
    Args:
        question: User's question
        context: Retrieved context text
        query_analysis: Optional QueryAnalysis object with query intelligence
        max_sentences: Maximum sentences to extract (default 2)
        enable_debug: Enable debug metadata in result
        
    Returns:
        FallbackResult with answer, confidence, and metadata
    """
    # Extract query intelligence
    if query_analysis is not None:
        query_type_str = query_analysis.query_type
        entities = query_analysis.entities if hasattr(query_analysis, 'entities') else []
        keywords = query_analysis.keywords if hasattr(query_analysis, 'keywords') else []
        comparison_targets = query_analysis.comparison_targets if hasattr(query_analysis, 'comparison_targets') else []
        temporal_markers = query_analysis.time_words if hasattr(query_analysis, 'time_words') else []
    else:
        # Fallback to basic extraction if no query analysis
        query_type_str = "factual"
        entities = []
        keywords = re.findall(r'\b\w+\b', question.lower())
        comparison_targets = []
        temporal_markers = []
    
    # Map query type string to enum
    try:
        query_type = QueryType(query_type_str)
    except ValueError:
        query_type = QueryType.FACTUAL
    
    # Handle unanswerable queries
    if query_type == QueryType.UNANSWERABLE:
        return FallbackResult(
            answer="I don't know from the provided context.",
            confidence=0.0,
            strategy="unanswerable_refusal",
            query_type=query_type_str,
            metadata={"reason": "Query detected as unanswerable"} if enable_debug else {}
        )
    
    # Split context into sentences
    sentences = split_sentences(context)
    
    if not sentences:
        return FallbackResult(
            answer="I don't know from the provided context.",
            confidence=0.0,
            strategy="no_context",
            query_type=query_type_str,
            metadata={"reason": "No sentences in context"} if enable_debug else {}
        )
    
    # Calculate base scores for all sentences
    base_scores = []
    for i, sentence in enumerate(sentences):
        # Keyword overlap
        keyword_score = calculate_keyword_overlap(sentence, keywords, entities)
        
        # Temporal boost
        temporal_boost = calculate_temporal_boost(sentence, temporal_markers, query_type)
        
        # Positional relevance
        positional_score = calculate_positional_relevance(
            sentence, i, len(sentences), query_type, question
        )
        
        # Answerability signals
        answerability_score = calculate_answerability_signals(sentence)
        
        # Combine scores with weights
        combined_score = (
            keyword_score * 0.4 +
            temporal_boost * 0.2 +
            positional_score * 0.2 +
            answerability_score * 0.2
        )
        
        base_scores.append(combined_score)
    
    # Select sentences based on query type strategy
    selected_sentences = []
    scored_sentences = []
    
    if query_type == QueryType.FACTUAL:
        selected_sentences, scored_sentences = select_factual_sentences(
            sentences, base_scores, keywords, entities, max_sentences
        )
        strategy = "factual_direct"
    
    elif query_type == QueryType.TEMPORAL:
        selected_sentences, scored_sentences = select_temporal_sentences(
            sentences, base_scores, question, max_sentences
        )
        strategy = "temporal_chronological"
    
    elif query_type == QueryType.COMPARATIVE:
        selected_sentences, scored_sentences = select_comparative_sentences(
            sentences, base_scores, comparison_targets, max_sentences
        )
        strategy = "comparative_balanced"
    
    elif query_type == QueryType.MULTI_HOP:
        # Use max_sentences=3 for multi-hop queries to capture entity + consequence
        multi_hop_max = 3
        selected_sentences, scored_sentences = select_multi_hop_sentences(
            sentences, base_scores, entities, question, multi_hop_max
        )
        strategy = "multi_hop_synthesis"
    
    elif query_type == QueryType.ANALYTICAL:
        selected_sentences, scored_sentences = select_analytical_sentences(
            sentences, base_scores, keywords, max_sentences
        )
        strategy = "analytical_cautious"
    
    elif query_type == QueryType.PROCEDURAL:
        selected_sentences, scored_sentences = select_procedural_sentences(
            sentences, base_scores, max_sentences
        )
        strategy = "procedural_sequential"
    
    else:
        # Default to factual
        selected_sentences, scored_sentences = select_factual_sentences(
            sentences, base_scores, keywords, entities, max_sentences
        )
        strategy = "default_factual"
    
    # Check if we have any valid selections
    if not selected_sentences:
        return FallbackResult(
            answer="I don't know from the provided context.",
            confidence=0.0,
            strategy="no_evidence",
            query_type=query_type_str,
            metadata={"reason": "No sentences selected"} if enable_debug else {}
        )
    
    # Calculate confidence based on best score
    best_score = max(score for sentence, score in scored_sentences[:len(selected_sentences)])
    confidence = min(best_score, 1.0)
    
    # Synthesize answer
    if query_type == QueryType.ANALYTICAL and confidence < 0.5:
        # Use cautious interpretive template for weak analytical evidence
        answer = generate_analytical_fallback(selected_sentences, keywords)
    else:
        answer = synthesize_answer(selected_sentences, query_type, question)
    
    # Build metadata if debug enabled
    metadata = {}
    if enable_debug:
        metadata = {
            "query_type": query_type_str,
            "entities": entities,
            "keywords": keywords,
            "comparison_targets": comparison_targets,
            "temporal_markers": temporal_markers,
            "total_sentences": len(sentences),
            "selected_count": len(selected_sentences),
        }
    
    return FallbackResult(
        answer=answer,
        confidence=confidence,
        strategy=strategy,
        selected_sentences=selected_sentences,
        sentence_scores=scored_sentences[:len(selected_sentences)],
        query_type=query_type_str,
        metadata=metadata,
    )


# ============================================================================
# LEGACY COMPATIBILITY WRAPPER
# ============================================================================

def extractive_fallback_legacy(
    context: str,
    question: str,
    return_confidence: bool = False,
) -> str | tuple[str, float]:
    """
    Legacy compatibility wrapper for the old extractive_fallback API.
    
    Maintains backward compatibility with existing code that expects
    the old signature (context, question) instead of (question, context).
    
    Args:
        context: Text context to search
        question: Question to match against
        return_confidence: If True, return (answer, confidence) tuple
        
    Returns:
        Answer string, or (answer, confidence) tuple if return_confidence=True
    """
    result = extractive_fallback(
        question=question,
        context=context,
        query_analysis=None,
        max_sentences=2,
        enable_debug=False,
    )
    
    if return_confidence:
        return result.answer, result.confidence
    return result.answer
