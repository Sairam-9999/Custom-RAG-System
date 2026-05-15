"""
Query Classification Module

This module provides a robust deterministic rule-based classification engine
for query intelligence in the RAG pipeline.

Architecture:
    Query → Rule Matching → Rule Scores → Conflict Resolution →
    Priority Resolution → Confidence Scoring → Query Type

The system is:
- Rule-based and deterministic (no LLM dependencies)
- Fully explainable with rich metadata
- Confidence-aware with scoring
- Conflict-resolving with priority ordering
- Domain-independent and production-safe
- Analytics-friendly with traceable decisions
"""

import re
from typing import Dict, Any, Tuple, List


# ============================================================================
# QUERY TYPE DEFINITIONS
# ============================================================================

QUERY_TYPES = [
    "factual",
    "reasoning",
    "comparative",
    "temporal",
    "procedural",
    "analytical",
    "multi_hop",
    "unanswerable",
]


PRIORITY_ORDER = [
    "unanswerable",
    "comparative",  # Comparative should beat multi_hop for explicit comparison queries
    "multi_hop",
    "temporal",
    "procedural",
    "reasoning",
    "analytical",
    "factual",
]


# ============================================================================
# RULE DEFINITIONS
# ============================================================================

FACTUAL_RULES = {
    "markers": [
        "who",
        "what",
        "when",
        "where",
        "which",
        "how many",
        "how much",
        "list",
        "name",
        "identify",
        "define",
        "describe",
    ],
    "patterns": [
        r"\bwhat is\b",
        r"\bwhat are\b",
        r"\bwhat was\b",
        r"\bwhat were\b",
    ],
    "weight": 1.0,
    "priority": 8,
}


REASONING_RULES = {
    "markers": [
        "why",
        "cause",
        "reason",
        "explain",
        "effect",
        "impact",
        "consequence",
        "result",
        "underlying",
        "mechanism",
    ],
    "patterns": [
        r"\bhow does\b",
        r"\bhow do\b",
    ],
    "weight": 1.1,
    "priority": 6,
}


COMPARATIVE_RULES = {
    "markers": [
        "compare",
        "vs",
        "versus",
        "difference",
        "better",
        "advantages",
        "disadvantages",
        "tradeoffs",
        "similarities",
        "contrast",
        "pros",
        "cons",
    ],
    "patterns": [
        r"\bcompared to\b",
        r"\brelative to\b",
        r"\bbetter than\b",
        r"\bworse than\b",
    ],
    "weight": 1.2,
    "priority": 3,
}


TEMPORAL_RULES = {
    "markers": [
        "before",
        "after",
        "during",
        "timeline",
        "sequence",
        "chronology",
        "eventually",
        "later",
        "first",
        "then",
        "next",
        "finally",
    ],
    "patterns": [
        r"\bwhen did\b",
        r"\bwhen was\b",
        r"\bhappened before\b",
        r"\bhappened after\b",
    ],
    "weight": 1.15,
    "priority": 4,
}


PROCEDURAL_RULES = {
    "markers": [
        "how to",
        "implement",
        "setup",
        "configure",
        "workflow",
        "process",
        "steps",
        "guide",
        "instruction",
        "create",
        "build",
        "deploy",
        "execute",
    ],
    "patterns": [
        r"\bhow do i\b",
        r"\bstep by step\b",
    ],
    "weight": 1.1,
    "priority": 5,
}


ANALYTICAL_RULES = {
    "markers": [
        "symbolize",
        "implication",
        "meaning",
        "mean",
        "interpret",
        "significance",
        "theme",
        "represents",
        "analyze",
        "evaluate",
        "assess",
        "critique",
        "perspective",
        "insight",
    ],
    "patterns": [
        r"\bwhat does.*mean\b",
        r"\bsymbolizes\b",
        r"\brepresents\b",
    ],
    "weight": 1.1,
    "priority": 7,
}


MULTI_HOP_RULES = {
    "markers": [],
    "patterns": [
        r"\bwho\b.*\bwhen\b",
        r"\bwhat\b.*\bwhen\b",
        r"\bwhy\b.*\bbefore\b",
        r"\bwhy\b.*\bafter\b",
        r"\bafter\b.*\bbut before\b",
        r"\bbefore\b.*\band after\b",
    ],
    "weight": 1.3,
    "priority": 2,
}


UNANSWERABLE_RULES = {
    "markers": [
        "it",
        "this",
        "that",
        "they",
        "them",
        "he",
        "she",
        "the thing",
        "the one",
        "there",
        "here",
        "something",
        "someone",
    ],
    "patterns": [
        r"^what happened there",
        r"^why was that",
        r"^who did it",
    ],
    "weight": 1.0,  # Reduced weight to lower confidence for unanswerable
    "priority": 1,
}


# ============================================================================
# RULE MATCHING FUNCTIONS
# ============================================================================

def match_factual_rules(query: str) -> Dict[str, Any]:
    """
    Match factual query rules.
    
    Returns structured rule match information.
    """
    query_lower = query.lower()
    matched_markers = []
    
    for marker in FACTUAL_RULES["markers"]:
        if marker in query_lower:
            matched_markers.append(marker)
    
    pattern_matches = 0
    for pattern in FACTUAL_RULES["patterns"]:
        if re.search(pattern, query_lower):
            pattern_matches += 1
    
    total_matches = len(matched_markers) + pattern_matches
    score = min(total_matches * FACTUAL_RULES["weight"] * 0.15, 0.95)
    
    return {
        "matched": total_matches > 0,
        "score": score,
        "matched_markers": matched_markers,
        "pattern_matches": pattern_matches,
        "reason": f"Detected {total_matches} factual markers/patterns",
    }


def match_reasoning_rules(query: str) -> Dict[str, Any]:
    """
    Match reasoning query rules.
    
    Returns structured rule match information.
    """
    query_lower = query.lower()
    matched_markers = []
    
    for marker in REASONING_RULES["markers"]:
        if marker in query_lower:
            matched_markers.append(marker)
    
    pattern_matches = 0
    for pattern in REASONING_RULES["patterns"]:
        if re.search(pattern, query_lower):
            pattern_matches += 1
    
    total_matches = len(matched_markers) + pattern_matches
    score = min(total_matches * REASONING_RULES["weight"] * 0.15, 0.95)
    
    return {
        "matched": total_matches > 0,
        "score": score,
        "matched_markers": matched_markers,
        "pattern_matches": pattern_matches,
        "reason": f"Detected {total_matches} reasoning markers/patterns",
    }


def match_comparative_rules(query: str) -> Dict[str, Any]:
    """
    Match comparative query rules.
    
    Returns structured rule match information.
    """
    query_lower = query.lower()
    matched_markers = []
    
    for marker in COMPARATIVE_RULES["markers"]:
        if marker in query_lower:
            matched_markers.append(marker)
    
    pattern_matches = 0
    for pattern in COMPARATIVE_RULES["patterns"]:
        if re.search(pattern, query_lower):
            pattern_matches += 1
    
    total_matches = len(matched_markers) + pattern_matches
    score = min(total_matches * COMPARATIVE_RULES["weight"] * 0.15, 0.95)
    
    return {
        "matched": total_matches > 0,
        "score": score,
        "matched_markers": matched_markers,
        "pattern_matches": pattern_matches,
        "reason": f"Detected {total_matches} comparative markers/patterns",
    }


def match_temporal_rules(query: str) -> Dict[str, Any]:
    """
    Match temporal query rules.
    
    Returns structured rule match information.
    """
    query_lower = query.lower()
    matched_markers = []
    
    for marker in TEMPORAL_RULES["markers"]:
        if marker in query_lower:
            matched_markers.append(marker)
    
    pattern_matches = 0
    for pattern in TEMPORAL_RULES["patterns"]:
        if re.search(pattern, query_lower):
            pattern_matches += 1
    
    total_matches = len(matched_markers) + pattern_matches
    score = min(total_matches * TEMPORAL_RULES["weight"] * 0.15, 0.95)
    
    return {
        "matched": total_matches > 0,
        "score": score,
        "matched_markers": matched_markers,
        "pattern_matches": pattern_matches,
        "reason": f"Detected {total_matches} temporal markers/patterns",
    }


def match_procedural_rules(query: str) -> Dict[str, Any]:
    """
    Match procedural query rules.
    
    Returns structured rule match information.
    """
    query_lower = query.lower()
    matched_markers = []
    
    for marker in PROCEDURAL_RULES["markers"]:
        if marker in query_lower:
            matched_markers.append(marker)
    
    pattern_matches = 0
    for pattern in PROCEDURAL_RULES["patterns"]:
        if re.search(pattern, query_lower):
            pattern_matches += 1
    
    total_matches = len(matched_markers) + pattern_matches
    score = min(total_matches * PROCEDURAL_RULES["weight"] * 0.15, 0.95)
    
    return {
        "matched": total_matches > 0,
        "score": score,
        "matched_markers": matched_markers,
        "pattern_matches": pattern_matches,
        "reason": f"Detected {total_matches} procedural markers/patterns",
    }


def match_analytical_rules(query: str) -> Dict[str, Any]:
    """
    Match analytical query rules.
    
    Returns structured rule match information.
    """
    query_lower = query.lower()
    matched_markers = []
    
    for marker in ANALYTICAL_RULES["markers"]:
        if marker in query_lower:
            matched_markers.append(marker)
    
    pattern_matches = 0
    for pattern in ANALYTICAL_RULES["patterns"]:
        if re.search(pattern, query_lower):
            pattern_matches += 1
    
    total_matches = len(matched_markers) + pattern_matches
    score = min(total_matches * ANALYTICAL_RULES["weight"] * 0.15, 0.95)
    
    return {
        "matched": total_matches > 0,
        "score": score,
        "matched_markers": matched_markers,
        "pattern_matches": pattern_matches,
        "reason": f"Detected {total_matches} analytical markers/patterns",
    }


def match_multi_hop_rules(query: str, entities: List[str]) -> Dict[str, Any]:
    """
    Match multi-hop query rules.
    
    Multi-hop queries require connecting multiple facts across entities.
    Detected via:
    - Multiple entities
    - Chained clauses with temporal/relational markers
    - Nested references
    """
    query_lower = query.lower()
    
    pattern_matches = 0
    for pattern in MULTI_HOP_RULES["patterns"]:
        if re.search(pattern, query_lower):
            pattern_matches += 1
    
    # Multi-hop detection: multiple entities + temporal/relational chaining
    temporal_markers = []
    temporal_word_list = ['before', 'after', 'during', 'later', 'earlier', 'first', 'next',
                         'then', 'eventually', 'timeline', 'sequence', 'chronology', 'when',
                         'while', 'until', 'since']
    for word in temporal_word_list:
        if word in query_lower:
            temporal_markers.append(word)
    
    relational_markers = ['when', 'after', 'before', 'while', 'during', 'then']
    has_relational = any(marker in query_lower for marker in relational_markers)
    
    # Check for chained clauses (multiple question words)
    question_words = ['who', 'what', 'why', 'how', 'when', 'where']
    qword_count = sum(1 for qw in question_words if qw in query_lower)
    
    is_multi_hop = (
        (len(entities) >= 2 and (has_relational or len(temporal_markers) > 0)) or
        pattern_matches > 0 or
        (qword_count >= 2 and len(entities) >= 1)
    )
    
    if is_multi_hop:
        score = min(
            (len(entities) * 0.1 + pattern_matches * 0.2 + has_relational * 0.15) * MULTI_HOP_RULES["weight"],
            0.95
        )
    else:
        score = 0.0
    
    return {
        "matched": is_multi_hop,
        "score": score,
        "matched_markers": [],
        "pattern_matches": pattern_matches,
        "entity_count": len(entities),
        "has_relational": has_relational,
        "reason": f"Multi-hop: {len(entities)} entities, {pattern_matches} patterns, relational={has_relational}",
    }


def match_unanswerable_rules(query: str, ambiguity_flags: List[str]) -> Dict[str, Any]:
    """
    Match unanswerable query rules.
    
    Unanswerable queries have vague pronouns, incomplete references,
    or missing concrete entities.
    
    Conservative approach: Only mark as unanswerable if:
    - Very short query with pronouns AND no strong content markers
    - Explicit vague patterns match
    """
    query_lower = query.lower()
    matched_markers = []
    
    for marker in UNANSWERABLE_RULES["markers"]:
        if marker in query_lower:
            matched_markers.append(marker)
    
    pattern_matches = 0
    for pattern in UNANSWERABLE_RULES["patterns"]:
        if re.search(pattern, query_lower):
            pattern_matches += 1
    
    # Strong content markers that indicate the query IS answerable
    strong_content_markers = [
        'significance', 'meaning', 'interpret', 'analyze', 'compare',
        'difference', 'explain', 'cause', 'reason', 'effect', 'impact',
        'implement', 'setup', 'configure', 'capital', 'wrote', 'created',
        'built', 'method', 'system', 'approach'
    ]
    
    has_strong_content = any(marker in query_lower for marker in strong_content_markers)
    
    # Unanswerable only if:
    # 1. Explicit pattern match, OR
    # 2. Very short with pronouns AND no strong content
    is_unanswerable = (
        pattern_matches > 0 or
        (len(query.split()) < 4 and len(matched_markers) > 0 and not has_strong_content)
    )
    
    if is_unanswerable:
        score = min(
            (len(ambiguity_flags) * 0.12 + len(matched_markers) * 0.08 + pattern_matches * 0.15) * UNANSWERABLE_RULES["weight"],
            0.85  # Cap unanswerable confidence lower
        )
    else:
        score = 0.0
    
    return {
        "matched": is_unanswerable,
        "score": score,
        "matched_markers": matched_markers,
        "pattern_matches": pattern_matches,
        "ambiguity_flags": ambiguity_flags,
        "reason": f"Unanswerable: {len(ambiguity_flags)} ambiguity flags, {len(matched_markers)} pronouns",
    }


# ============================================================================
# CONFLICT RESOLUTION
# ============================================================================

def resolve_conflicts(rule_matches: Dict[str, Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    """
    Resolve conflicts between matched rules using priority ordering.
    
    Args:
        rule_matches: Dictionary mapping query types to their match results
        
    Returns:
        Tuple of (winning_query_type, conflict_metadata)
    """
    # Filter to only matched rules
    matched_types = {qt: rm for qt, rm in rule_matches.items() if rm["matched"]}
    
    if not matched_types:
        # No matches - default to factual
        return "factual", {
            "reason": "No rules matched, defaulting to factual",
            "conflicting_types": [],
        }
    
    # If only one type matched, no conflict
    if len(matched_types) == 1:
        query_type = list(matched_types.keys())[0]
        return query_type, {
            "reason": f"Single match: {query_type}",
            "conflicting_types": [],
        }
    
    # Multiple matches - use priority ordering
    # Higher priority (lower number) wins
    priorities = {qt: PRIORITY_ORDER.index(qt) for qt in matched_types.keys()}
    
    # Sort by priority (lower index = higher priority)
    sorted_by_priority = sorted(priorities.items(), key=lambda x: x[1])
    
    # Get the highest priority type
    winner = sorted_by_priority[0][0]
    
    # Check if scores are close (within 0.1)
    winner_score = matched_types[winner]["score"]
    close_competitors = [
        qt for qt, rm in matched_types.items()
        if qt != winner and abs(rm["score"] - winner_score) < 0.1
    ]
    
    conflict_metadata = {
        "priority_winner": winner,
        "winner_priority": priorities[winner],
        "winner_score": winner_score,
        "conflicting_types": list(matched_types.keys()),
        "close_competitors": close_competitors,
        "reason": f"Priority resolution: {winner} (priority {priorities[winner]}) won over {len(matched_types)-1} other types",
    }
    
    return winner, conflict_metadata


# ============================================================================
# CONFIDENCE SCORING
# ============================================================================

def calculate_confidence(
    rule_matches: Dict[str, Dict[str, Any]],
    winner: str,
    conflict_metadata: Dict[str, Any],
) -> float:
    """
    Calculate confidence score for the classification.
    
    Confidence depends on:
    - Strength of winning rule score
    - Number of matched markers
    - Query clarity (length, entity count)
    - Ambiguity penalties
    - Competing rule overlap
    """
    winner_match = rule_matches[winner]
    base_score = winner_match["score"]
    
    # Ensure base_score is a float
    if not isinstance(base_score, (int, float)):
        base_score = float(base_score) if base_score else 0.5
    
    # Boost for strong marker matches
    matched_markers = winner_match.get("matched_markers", [])
    marker_boost = min(len(matched_markers) * 0.05, 0.2)
    
    # Penalty for close competitors (ambiguity)
    close_competitors = conflict_metadata.get("close_competitors", [])
    competitor_penalty = len(close_competitors) * 0.1
    
    # Penalty for ambiguity flags
    ambiguity_flags = winner_match.get("ambiguity_flags", [])
    ambiguity_penalty = len(ambiguity_flags) * 0.15
    
    # Calculate final confidence
    confidence = base_score + marker_boost - competitor_penalty - ambiguity_penalty
    
    # Clamp to [0.35, 0.95] range
    confidence = max(0.35, min(0.95, confidence))
    
    return confidence


# ============================================================================
# MAIN CLASSIFICATION ENGINE
# ============================================================================

def classify_query(query: str, entities: List[str], ambiguity_flags: List[str], debug: bool = False) -> Tuple[str, float, Dict[str, Any]]:
    """
    Classify query using the rule-based scoring engine.
    
    Architecture:
        Query → Rule Matching → Rule Scores → Conflict Resolution →
        Priority Resolution → Confidence Scoring → Query Type
    
    Args:
        query: The user query string
        entities: Extracted entities from the query
        ambiguity_flags: Detected ambiguity flags
        debug: If True, return debug metadata
        
    Returns:
        Tuple of (query_type, confidence, debug_metadata)
    """
    # Step 1: Match all rules
    rule_matches = {
        "factual": match_factual_rules(query),
        "reasoning": match_reasoning_rules(query),
        "comparative": match_comparative_rules(query),
        "temporal": match_temporal_rules(query),
        "procedural": match_procedural_rules(query),
        "analytical": match_analytical_rules(query),
        "multi_hop": match_multi_hop_rules(query, entities),
        "unanswerable": match_unanswerable_rules(query, ambiguity_flags),
    }
    
    # Step 2: Resolve conflicts
    winner, conflict_metadata = resolve_conflicts(rule_matches)
    
    # Step 3: Calculate confidence
    confidence = calculate_confidence(rule_matches, winner, conflict_metadata)
    
    # Step 4: Build debug metadata
    debug_metadata = {
        "matched_rules": [qt for qt, rm in rule_matches.items() if rm["matched"]],
        "rule_scores": {qt: rm["score"] for qt, rm in rule_matches.items()},
        "matched_markers": {
            qt: rm.get("matched_markers", []) for qt, rm in rule_matches.items()
        },
        **conflict_metadata,
    }
    
    if debug:
        return winner, confidence, debug_metadata
    return winner, confidence, {}


def get_query_type_description(query_type: str) -> str:
    """
    Get human-readable description of a query type.
    
    Args:
        query_type: The query type string
        
    Returns:
        Description string
    """
    descriptions = {
        'factual': 'Direct information retrieval requiring minimal reasoning',
        'reasoning': 'Questions requiring causal explanation or logical reasoning',
        'comparative': 'Questions comparing multiple entities or concepts',
        'temporal': 'Questions involving time, sequence, or chronology',
        'procedural': 'How-to or process-oriented questions',
        'analytical': 'Questions requiring deep analysis or interpretation',
        'multi_hop': 'Questions requiring multiple interconnected retrieval steps',
        'unanswerable': 'Vague or incomplete questions lacking context',
    }
    return descriptions.get(query_type, 'Unknown query type')


__all__ = [
    'QUERY_TYPES',
    'PRIORITY_ORDER',
    'FACTUAL_RULES',
    'REASONING_RULES',
    'COMPARATIVE_RULES',
    'TEMPORAL_RULES',
    'PROCEDURAL_RULES',
    'ANALYTICAL_RULES',
    'MULTI_HOP_RULES',
    'UNANSWERABLE_RULES',
    'match_factual_rules',
    'match_reasoning_rules',
    'match_comparative_rules',
    'match_temporal_rules',
    'match_procedural_rules',
    'match_analytical_rules',
    'match_multi_hop_rules',
    'match_unanswerable_rules',
    'resolve_conflicts',
    'calculate_confidence',
    'classify_query',
    'get_query_type_description',
]
