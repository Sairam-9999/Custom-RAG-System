"""
Slot Extraction Module

This module provides entity coverage detection and semantic answer target extraction.
It separates anchor entities from answer slots and supports generic query forms.
"""

import re
from typing import List, Optional, Tuple, Any

from ..core.types import AnswerTarget
from ..core.config import QUERY_NORMALIZATION_CONFIG


def detect_entity_coverage_need(query: str, query_type: str, entities: List[str]) -> bool:
    """
    Detect if a query requires evidence for multiple answer slots.
    
    This function identifies queries that ask for multiple pieces of information,
    entities, attributes, or answers that require separate evidence coverage.
    
    Args:
        query: The user query string
        query_type: The classified query type
        entities: Extracted entities from the query
        
    Returns:
        True if the query requires entity coverage, False otherwise
    """
    query_lower = query.lower()
    
    # Multi-conjunction patterns indicating multiple slots
    multi_slot_patterns = [
        r'\band\b.*\band\b',  # "X and Y and Z"
        r'\bor\b.*\bor\b',    # "X or Y or Z"
        r'who are\s+',        # "Who are X and Y?"
        r'what are\s+',       # "What are X and Y?"
        r'list the\s+',       # "List the characters..."
        r'names of\s+',       # "names of X and Y"
        r'identify\s+.*\s+and\s+',  # "identify X and Y"
    ]
    
    for pattern in multi_slot_patterns:
        if re.search(pattern, query_lower):
            return True
    
    # Multiple entities in factual queries
    if query_type == "factual" and len(entities) >= 2:
        # Check if entities are connected by "and" or listed
        if re.search(r'\band\b', query_lower):
            return True
    
    # Comparative queries always need entity coverage
    if query_type == "comparative":
        return True
    
    # Multi-hop queries with multiple entities need coverage
    if query_type == "multi_hop" and len(entities) >= 2:
        return True
    
    # Queries asking for multiple attributes
    multi_attribute_patterns = [
        r'advantages and disadvantages',
        r'pros and cons',
        r'causes and effects',
        r'benefits and drawbacks',
        r'strengths and weaknesses',
    ]
    
    for pattern in multi_attribute_patterns:
        if re.search(pattern, query_lower):
            return True
    
    # Questions with multiple question words
    question_words = ['who', 'what', 'when', 'where', 'why', 'how']
    qword_count = sum(1 for qw in question_words if qw in query_lower)
    if qword_count >= 2:
        return True
    
    # Queries with numbered lists or enumerations
    if re.search(r'(?:first|second|third|1\.|2\.|3\.)', query_lower):
        return True
    
    return False


def normalize_anchor_entity(anchor: str, known_entities: Optional[List[str]] = None) -> str:
    """
    Normalize anchor entity using fuzzy matching to handle typos.
    
    This function corrects common typos in entity names using fuzzy matching
    against known entities from the document or a predefined alias map.
    
    Args:
        anchor: The anchor entity string to normalize
        known_entities: Optional list of known entities from the document
        
    Returns:
        Normalized anchor entity string
    """
    if known_entities:
        import difflib
        # Use fuzzy matching against known entities
        matches = difflib.get_close_matches(anchor.lower(), [e.lower() for e in known_entities], n=1, cutoff=0.75)
        if matches:
            # Return the original casing of the matched entity
            for entity in known_entities:
                if entity.lower() == matches[0]:
                    return entity
    
    # Optional application-provided aliases. Default config is intentionally empty
    # so the core library remains domain-independent.
    anchor_lower = anchor.lower()
    configured_alias = QUERY_NORMALIZATION_CONFIG.entity_aliases.get(anchor_lower)
    if configured_alias:
        return configured_alias

    return anchor


def extract_expected_answer_slots(query: str, query_type: str, entities: List[str]) -> Tuple[Optional[str], List[str]]:
    """
    Extract expected answer slots from a query.
    
    This function identifies the specific answer slots that need evidence coverage
    and separates them from the anchor entity.
    For example, "What are Cooper's son and daughter named?" returns ("Cooper", ["son", "daughter"]).
    
    Args:
        query: The user query string
        query_type: The classified query type
        entities: Extracted entities from the query
        
    Returns:
        Tuple of (anchor_entity, list of expected answer slots)
    """
    query_lower = query.lower()
    slots = []
    anchor_entity = None

    # Generic relation expansion, configured centrally.
    # Example: category term -> concrete slots, such as children -> [son, daughter].
    for relation, expanded_slots in QUERY_NORMALIZATION_CONFIG.relation_expansions.items():
        category_pattern = rf"(?:names? of\s+)?(\w+)\s+{re.escape(relation)}\??$"
        match = re.search(category_pattern, query_lower)
        if match:
            anchor_entity = normalize_anchor_entity(match.group(1), entities)
            return anchor_entity, list(expanded_slots)
    
    # For comparative queries, use comparison targets as slots
    if query_type == "comparative":
        from .parser import extract_comparison_targets
        comparison_targets = extract_comparison_targets(query)
        if comparison_targets:
            return None, comparison_targets
    
    # Pattern: "What are X's A and B named?" or similar - check this BEFORE generic "what are"
    # Examples: "What are Cooper's son and daughter named?" -> anchor="Cooper", slots=["son", "daughter"]
    what_are_possessive_pattern = r"what are\s+(\w+)[’']s\s+(\w+)\s+and\s+(\w+)"
    match = re.search(what_are_possessive_pattern, query_lower)
    if match:
        anchor_entity = match.group(1)
        slots.extend([match.group(2), match.group(3)])
        anchor_entity = normalize_anchor_entity(anchor_entity, entities)
        return anchor_entity, slots

    # Pattern: "Who are X's A and B?" or similar
    who_are_possessive_pattern = r"who are\s+(\w+)[’']s\s+(\w+)\s+and\s+(\w+)"
    match = re.search(who_are_possessive_pattern, query_lower)
    if match:
        anchor_entity = match.group(1)
        slots.extend([match.group(2), match.group(3)])
        anchor_entity = normalize_anchor_entity(anchor_entity, entities)
        return anchor_entity, slots

    # Pattern: "X's A and B" in possessive or relationship context
    # Examples: "Cooper's son and daughter" -> anchor="Cooper", slots=["son", "daughter"]
    possessive_pattern = r"(\w+)[’']s\s+(\w+)\s+and\s+(\w+)"
    match = re.search(possessive_pattern, query_lower)
    if match:
        anchor_entity = match.group(1)
        slots.extend([match.group(2), match.group(3)])
        # Normalize anchor entity
        anchor_entity = normalize_anchor_entity(anchor_entity, entities)
        return anchor_entity, slots

    # Pattern: "names of X's A and B"
    names_of_pattern = r"names?\s+of\s+(\w+)[’']s\s+(\w+)\s+and\s+(\w+)"
    match = re.search(names_of_pattern, query_lower)
    if match:
        anchor_entity = match.group(1)
        slots.extend([match.group(2), match.group(3)])
        anchor_entity = normalize_anchor_entity(anchor_entity, entities)
        return anchor_entity, slots

    # Pattern: "what are the names of X A and B" (non-possessive)
    # Examples: "what are the names of cooper son and daughter" -> anchor="cooper", slots=["son", "daughter"]
    what_are_names_of_pattern = r"what are the names of\s+(\w+)\s+(\w+)\s+and\s+(\w+)"
    match = re.search(what_are_names_of_pattern, query_lower)
    if match:
        anchor_entity = match.group(1)
        slots.extend([match.group(2), match.group(3)])
        anchor_entity = normalize_anchor_entity(anchor_entity, entities)
        return anchor_entity, slots

    # Pattern: "what are the names of X A and B" (with "the")
    what_are_names_of_pattern_alt = r"what are names of\s+(\w+)\s+(\w+)\s+and\s+(\w+)"
    match = re.search(what_are_names_of_pattern_alt, query_lower)
    if match:
        anchor_entity = match.group(1)
        slots.extend([match.group(2), match.group(3)])
        anchor_entity = normalize_anchor_entity(anchor_entity, entities)
        return anchor_entity, slots

    # Pattern: "who are X A and B" (non-possessive)
    # Examples: "who are cooper son and daughter" -> anchor="cooper", slots=["son", "daughter"]
    who_are_non_possessive_pattern = r"who are\s+(\w+)\s+(\w+)\s+and\s+(\w+)"
    match = re.search(who_are_non_possessive_pattern, query_lower)
    if match:
        anchor_entity = match.group(1)
        slots.extend([match.group(2), match.group(3)])
        anchor_entity = normalize_anchor_entity(anchor_entity, entities)
        return anchor_entity, slots

    # Pattern: "name X A and B" (non-possessive)
    # Examples: "name cooper son and daughter" -> anchor="cooper", slots=["son", "daughter"]
    name_non_possessive_pattern = r"name\s+(\w+)\s+(\w+)\s+and\s+(\w+)"
    match = re.search(name_non_possessive_pattern, query_lower)
    if match:
        anchor_entity = match.group(1)
        slots.extend([match.group(2), match.group(3)])
        anchor_entity = normalize_anchor_entity(anchor_entity, entities)
        return anchor_entity, slots

    # Pattern: "X A and B" (generic anchor + relation, non-possessive)
    # Examples: "cooper son and daughter" -> anchor="cooper", slots=["son", "daughter"]
    # This pattern matches when we have three words in sequence without possessive or question words
    generic_anchor_relation_pattern = r"^(\w+)\s+(\w+)\s+and\s+(\w+)$"
    match = re.search(generic_anchor_relation_pattern, query_lower.strip())
    if match:
        # Only apply if this looks like an entity + relationship pattern
        # Check if the first word is likely an entity (capitalized in original query)
        first_word = match.group(1)
        original_query_words = query.split()
        if original_query_words and original_query_words[0][0].isupper():
            anchor_entity = match.group(1)
            slots.extend([match.group(2), match.group(3)])
            anchor_entity = normalize_anchor_entity(anchor_entity, entities)
            return anchor_entity, slots

    # Pattern: "names of A and B" (without possessive)
    names_of_pattern_no_possessive = r"names?\s+of\s+(.+?)\s+and\s+(.+?)(?:\s|$|\.|\?|!)"
    match = re.search(names_of_pattern_no_possessive, query_lower)
    if match:
        # When there's no clear possessive, treat the first part as context
        # and extract the actual slots
        slots.extend([match.group(1).strip(), match.group(2).strip()])
        return None, slots

    # Pattern: "Who are X and Y?"
    who_are_pattern = r"who are\s+(.+?)\s+and\s+(.+?)(?:\?|$)"
    match = re.search(who_are_pattern, query_lower)
    if match:
        slots.extend([match.group(1).strip(), match.group(2).strip()])
        return None, slots

    # Pattern: "What are X and Y?"
    what_are_pattern = r"what are\s+(.+?)\s+and\s+(.+?)(?:\?|$)"
    match = re.search(what_are_pattern, query_lower)
    if match:
        slots.extend([match.group(1).strip(), match.group(2).strip()])
        return None, slots
    
    # Pattern: "List the X, Y, and Z"
    list_pattern = r"list\s+the\s+(.+?)(?:,|\s+and\s+)"
    match = re.search(list_pattern, query_lower)
    if match:
        # Extract comma-separated items after "list the"
        rest_of_query = query_lower[match.end():]
        items = re.split(r'\s*,\s*|\s+and\s+', rest_of_query)
        items = [item.strip().rstrip('.?!') for item in items if item.strip()]
        slots.extend(items[:3])  # Limit to first 3 items
        return None, slots
    
    # Pattern: "advantages and disadvantages"
    if 'advantages' in query_lower and 'disadvantages' in query_lower:
        slots.extend(['advantages', 'disadvantages'])
        return None, slots
    
    if 'pros' in query_lower and 'cons' in query_lower:
        slots.extend(['pros', 'cons'])
        return None, slots
    
    if 'causes' in query_lower and 'effects' in query_lower:
        slots.extend(['causes', 'effects'])
        return None, slots
    
    # For multi-hop queries with entities, use entities as slots
    if query_type == "multi_hop" and len(entities) >= 2:
        return None, entities[:3]  # Limit to first 3 entities
    
    # Fallback: extract entities connected by "and"
    if len(entities) >= 2:
        # Find entities that appear in an "and" pattern
        and_pattern = r'(\w+(?:\s+\w+)*)\s+and\s+(\w+(?:\s+\w+)*)'
        matches = re.findall(and_pattern, query_lower)
        for match in matches:
            slots.extend([match[0].strip(), match[1].strip()])
        if slots:
            return None, slots
    
    # Last resort: use all entities as slots
    if entities:
        return None, entities[:3]
    
    return None, slots


def build_answer_targets(query: str, query_type: str, entities: List[str]) -> Tuple[Optional[str], List[AnswerTarget]]:
    """
    Build semantic AnswerTarget objects from a query.
    
    This function performs semantic extraction of answer targets, properly
    separating anchor entities from target types and requested attributes.
    It supports generic query forms without domain-specific hardcoding.
    
    Args:
        query: The user query string
        query_type: The classified query type
        entities: Extracted entities from the query
        
    Returns:
        Tuple of (anchor_entity, list of AnswerTarget objects)
        
    Examples:
        Query: "What are the names of Entity's relation_a and relation_b?"
        Returns: ("cooper", [
            AnswerTarget(anchor_entity="cooper", target_type="son", requested_attribute="name"),
            AnswerTarget(anchor_entity="cooper", target_type="daughter", requested_attribute="name")
        ])
        
        Query: "Compare BM25 vs dense retrieval"
        Returns: (None, [
            AnswerTarget(anchor_entity=None, target_type="BM25", requested_attribute=None),
            AnswerTarget(anchor_entity=None, target_type="dense retrieval", requested_attribute=None)
        ])
    """
    query_lower = query.lower()
    answer_targets = []
    anchor_entity = None
    
    # Detect requested attribute (name, age, location, etc.)
    requested_attribute = None
    attribute_patterns = [
        r'\b(?:named?|names?)\b',
        r'\b(?:called|call)\b',
        r'\b(?:age|how old)\b',
        r'\b(?:location|where|located)\b',
        r'\b(?:role|function|purpose)\b',
    ]
    for pattern in attribute_patterns:
        if re.search(pattern, query_lower):
            if 'age' in pattern or 'how old' in pattern:
                requested_attribute = 'age'
            elif 'location' in pattern or 'where' in pattern or 'located' in pattern:
                requested_attribute = 'location'
            elif 'role' in pattern or 'function' in pattern or 'purpose' in pattern:
                requested_attribute = 'role'
            else:
                requested_attribute = 'name'
            break
    
    # Pattern 1: "what are the names of X A and B" (non-possessive)
    # Examples: "what are the names of cooper son and daughter" -> anchor="cooper", slots=["son", "daughter"]
    what_are_names_of_pattern = r"what are the names of\s+(\w+)\s+(\w+)\s+and\s+(\w+)"
    match = re.search(what_are_names_of_pattern, query_lower)
    if match:
        anchor_entity = normalize_anchor_entity(match.group(1), entities)
        answer_targets.append(AnswerTarget(
            anchor_entity=anchor_entity,
            target_type=match.group(2),
            requested_attribute=requested_attribute or 'name'
        ))
        answer_targets.append(AnswerTarget(
            anchor_entity=anchor_entity,
            target_type=match.group(3),
            requested_attribute=requested_attribute or 'name'
        ))
        return anchor_entity, answer_targets

    # Pattern 2: "what are names of X A and B" (non-possessive, without "the")
    what_are_names_of_pattern_alt = r"what are names of\s+(\w+)\s+(\w+)\s+and\s+(\w+)"
    match = re.search(what_are_names_of_pattern_alt, query_lower)
    if match:
        anchor_entity = normalize_anchor_entity(match.group(1), entities)
        answer_targets.append(AnswerTarget(
            anchor_entity=anchor_entity,
            target_type=match.group(2),
            requested_attribute=requested_attribute or 'name'
        ))
        answer_targets.append(AnswerTarget(
            anchor_entity=anchor_entity,
            target_type=match.group(3),
            requested_attribute=requested_attribute or 'name'
        ))
        return anchor_entity, answer_targets

    # Pattern 3: "who are X A and B" (non-possessive)
    # Examples: "who are cooper son and daughter" -> anchor="cooper", slots=["son", "daughter"]
    who_are_non_possessive_pattern = r"who are\s+(\w+)\s+(\w+)\s+and\s+(\w+)"
    match = re.search(who_are_non_possessive_pattern, query_lower)
    if match:
        anchor_entity = normalize_anchor_entity(match.group(1), entities)
        answer_targets.append(AnswerTarget(
            anchor_entity=anchor_entity,
            target_type=match.group(2),
            requested_attribute=requested_attribute or 'name'
        ))
        answer_targets.append(AnswerTarget(
            anchor_entity=anchor_entity,
            target_type=match.group(3),
            requested_attribute=requested_attribute or 'name'
        ))
        return anchor_entity, answer_targets

    # Pattern 4: "name X A and B" (non-possessive)
    # Examples: "name cooper son and daughter" -> anchor="cooper", slots=["son", "daughter"]
    name_non_possessive_pattern = r"name\s+(\w+)\s+(\w+)\s+and\s+(\w+)"
    match = re.search(name_non_possessive_pattern, query_lower)
    if match:
        anchor_entity = normalize_anchor_entity(match.group(1), entities)
        answer_targets.append(AnswerTarget(
            anchor_entity=anchor_entity,
            target_type=match.group(2),
            requested_attribute=requested_attribute or 'name'
        ))
        answer_targets.append(AnswerTarget(
            anchor_entity=anchor_entity,
            target_type=match.group(3),
            requested_attribute=requested_attribute or 'name'
        ))
        return anchor_entity, answer_targets

    # Pattern 5: "X A and B" (generic anchor + relation, non-possessive)
    # Examples: "cooper son and daughter" -> anchor="cooper", slots=["son", "daughter"]
    generic_anchor_relation_pattern = r"^(\w+)\s+(\w+)\s+and\s+(\w+)$"
    match = re.search(generic_anchor_relation_pattern, query_lower.strip())
    if match:
        # Only apply if this looks like an entity + relationship pattern
        # Check if the first word is likely an entity (capitalized in original query)
        original_query_words = query.split()
        if original_query_words and original_query_words[0][0].isupper():
            anchor_entity = normalize_anchor_entity(match.group(1), entities)
            answer_targets.append(AnswerTarget(
                anchor_entity=anchor_entity,
                target_type=match.group(2),
                requested_attribute=requested_attribute
            ))
            answer_targets.append(AnswerTarget(
                anchor_entity=anchor_entity,
                target_type=match.group(3),
                requested_attribute=requested_attribute
            ))
            return anchor_entity, answer_targets

    # Pattern 6: Possessive queries with explicit "named" or "names"
    # "What are the names of X's A and B?"
    possessive_named_pattern = r"names?\s+(?:of\s+)?(\w+)[’']s\s+(\w+)\s+and\s+(\w+)"
    match = re.search(possessive_named_pattern, query_lower)
    if match:
        anchor_entity = normalize_anchor_entity(match.group(1), entities)
        answer_targets.append(AnswerTarget(
            anchor_entity=anchor_entity,
            target_type=match.group(2),
            requested_attribute=requested_attribute or 'name'
        ))
        answer_targets.append(AnswerTarget(
            anchor_entity=anchor_entity,
            target_type=match.group(3),
            requested_attribute=requested_attribute or 'name'
        ))
        return anchor_entity, answer_targets
    
    # Pattern 2: Possessive queries without explicit "named"
    # "X's A and B"
    possessive_pattern = r"(\w+)[’']s\s+(\w+)\s+and\s+(\w+)"
    match = re.search(possessive_pattern, query_lower)
    if match:
        anchor_entity = normalize_anchor_entity(match.group(1), entities)
        answer_targets.append(AnswerTarget(
            anchor_entity=anchor_entity,
            target_type=match.group(2),
            requested_attribute=requested_attribute
        ))
        answer_targets.append(AnswerTarget(
            anchor_entity=anchor_entity,
            target_type=match.group(3),
            requested_attribute=requested_attribute
        ))
        return anchor_entity, answer_targets
    
    # Pattern 3: "names of A and B" (non-possessive)
    names_of_pattern = r"names?\s+of\s+(.+?)\s+and\s+(.+?)(?:\s|$|\.|\?|!)"
    match = re.search(names_of_pattern, query_lower)
    if match:
        # No clear anchor entity, treat both as independent targets
        answer_targets.append(AnswerTarget(
            anchor_entity=None,
            target_type=match.group(1).strip(),
            requested_attribute=requested_attribute or 'name'
        ))
        answer_targets.append(AnswerTarget(
            anchor_entity=None,
            target_type=match.group(2).strip(),
            requested_attribute=requested_attribute or 'name'
        ))
        return None, answer_targets
    
    # Pattern 4: Comparative queries
    if query_type == "comparative":
        from .parser import extract_comparison_targets
        comparison_targets = extract_comparison_targets(query)
        if comparison_targets:
            for target in comparison_targets:
                answer_targets.append(AnswerTarget(
                    anchor_entity=None,
                    target_type=target,
                    requested_attribute=None
                ))
            return None, answer_targets
    
    # Pattern 5: "Who are X and Y?"
    who_are_pattern = r"who are\s+(.+?)\s+and\s+(.+?)(?:\?|$)"
    match = re.search(who_are_pattern, query_lower)
    if match:
        answer_targets.append(AnswerTarget(
            anchor_entity=None,
            target_type=match.group(1).strip(),
            requested_attribute='identity'
        ))
        answer_targets.append(AnswerTarget(
            anchor_entity=None,
            target_type=match.group(2).strip(),
            requested_attribute='identity'
        ))
        return None, answer_targets
    
    # Pattern 6: "What are X and Y?"
    what_are_pattern = r"what are\s+(.+?)\s+and\s+(.+?)(?:\?|$)"
    match = re.search(what_are_pattern, query_lower)
    if match:
        answer_targets.append(AnswerTarget(
            anchor_entity=None,
            target_type=match.group(1).strip(),
            requested_attribute=None
        ))
        answer_targets.append(AnswerTarget(
            anchor_entity=None,
            target_type=match.group(2).strip(),
            requested_attribute=None
        ))
        return None, answer_targets
    
    # Pattern 7: List queries
    list_pattern = r"list\s+(?:the\s+)?(.+?)(?:,|\s+and\s+)"
    match = re.search(list_pattern, query_lower)
    if match:
        rest_of_query = query_lower[match.end():]
        items = re.split(r'\s*,\s*|\s+and\s+', rest_of_query)
        items = [item.strip().rstrip('.?!') for item in items if item.strip()]
        for item in items[:3]:
            answer_targets.append(AnswerTarget(
                anchor_entity=None,
                target_type=item,
                requested_attribute=None
            ))
        return None, answer_targets
    
    # Pattern 8: Multi-concept queries (advantages/disadvantages, pros/cons, causes/effects)
    concept_pairs = [
        (['advantages', 'disadvantages'], None),
        (['pros', 'cons'], None),
        (['causes', 'effects'], None),
    ]
    for pair, attr in concept_pairs:
        if all(term in query_lower for term in pair):
            for term in pair:
                answer_targets.append(AnswerTarget(
                    anchor_entity=None,
                    target_type=term,
                    requested_attribute=attr
                ))
            return None, answer_targets
    
    # Pattern 9: Multi-hop queries with entities
    if query_type == "multi_hop" and len(entities) >= 2:
        for entity in entities[:3]:
            answer_targets.append(AnswerTarget(
                anchor_entity=None,
                target_type=entity,
                requested_attribute=None
            ))
        return None, answer_targets
    
    # Fallback: Extract entities connected by "and"
    if len(entities) >= 2:
        and_pattern = r'(\w+(?:\s+\w+)*)\s+and\s+(\w+(?:\s+\w+)*)'
        matches = re.findall(and_pattern, query_lower)
        for match in matches:
            answer_targets.append(AnswerTarget(
                anchor_entity=None,
                target_type=match[0].strip(),
                requested_attribute=None
            ))
            answer_targets.append(AnswerTarget(
                anchor_entity=None,
                target_type=match[1].strip(),
                requested_attribute=None
            ))
        if answer_targets:
            return None, answer_targets
    
    # Last resort: use entities as targets
    if entities:
        for entity in entities[:3]:
            answer_targets.append(AnswerTarget(
                anchor_entity=None,
                target_type=entity,
                requested_attribute=None
            ))
        return None, answer_targets
    
    return None, answer_targets


def extract_slot_answers_from_evidence(
    chunks: List[str],
    expected_answer_slots: List[str],
    anchor_entity: Optional[str] = None
) -> dict:
    """
    Deterministically extract slot answers from retrieved evidence.
    
    This function parses retrieved chunks to extract specific slot values
    before relying on LLM generation. It uses regex patterns to find
    direct evidence for each slot.
    
    Args:
        chunks: List of retrieved text chunks
        expected_answer_slots: List of slot names to extract (e.g., ['son', 'daughter'])
        anchor_entity: Optional anchor entity to help with extraction (e.g., 'Cooper')
    
    Returns:
        Dict mapping slot names to extracted values (None if not found)
    """
    combined_text = " ".join(chunks)
    results = {slot: None for slot in expected_answer_slots}
    
    for slot in expected_answer_slots:
        # Pattern 1: "NAME, anchor_entity's ... slot" (enhanced)
        # Example pattern: "NAME, Entity’s relation" -> relation = Name
        if anchor_entity:
            # Process each chunk individually to avoid cross-matching
            for chunk in chunks:
                pattern1 = rf"([A-Z][A-Za-z]+),\s+{anchor_entity}['']s[^.]*?{slot}"
                match = re.search(pattern1, chunk, re.IGNORECASE)
                if match:
                    # Normalize casing for extracted names.
                    results[slot] = match.group(1).title()
                    break  # Found in this chunk, move to next slot
            if results[slot] is not None:
                continue
        
        # Pattern 2: "slot. This is NAME"
        # Example pattern: "relation. This is NAME" -> relation = Name
        pattern2 = rf"{slot}\.?\s+(?:This is|named|called)\s+([A-Z][A-Za-z]+)"
        match = re.search(pattern2, combined_text, re.IGNORECASE)
        if match:
            results[slot] = match.group(1)
            continue
        
        # Pattern 3: "slot: NAME" or "slot is NAME"
        pattern3 = rf"{slot}\s*(?:is|:)\s+([A-Z][A-Za-z]+)"
        match = re.search(pattern3, combined_text, re.IGNORECASE)
        if match:
            results[slot] = match.group(1)
            continue
        
        # Pattern 4: "anchor_entity's slot is NAME"
        if anchor_entity:
            pattern4 = rf"{anchor_entity}'s\s+{slot}\s+(?:is|was)\s+([A-Z][A-Za-z]+)"
            match = re.search(pattern4, combined_text, re.IGNORECASE)
            if match:
                results[slot] = match.group(1)
                continue
        
        # Pattern 5: "NAME is anchor_entity's slot"
        if anchor_entity:
            pattern5 = rf"([A-Z][A-Za-z]+)\s+(?:is|was)\s+{anchor_entity}'s\s+{slot}"
            match = re.search(pattern5, combined_text, re.IGNORECASE)
            if match:
                results[slot] = match.group(1)
                continue
    
    return results


__all__ = [
    'detect_entity_coverage_need',
    'normalize_anchor_entity',
    'extract_expected_answer_slots',
    'build_answer_targets',
    'extract_slot_answers_from_evidence',
]
