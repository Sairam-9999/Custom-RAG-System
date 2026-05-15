"""
Query Parsing Module

This module provides utilities for parsing queries to extract:
- Entities
- Keywords
- Temporal markers
- Comparison targets
- Question focus
- Ambiguity flags

These parsing functions are domain-independent and use heuristic-based extraction.
"""

import re
from typing import List, Optional


def extract_entities(query: str) -> List[str]:
    """
    Extract potential entities from the query.
    
    This function uses heuristics to identify capitalized phrases,
    quoted strings, single-letter entities, and other potential entity mentions.
    
    Args:
        query: The user query string
        
    Returns:
        List of extracted entity strings
    """
    entities = []
    
    # Extract quoted strings
    quoted_pattern = r'"([^"]+)"'
    entities.extend(re.findall(quoted_pattern, query))
    
    # Extract comparison targets first (A vs B pattern)
    # Match only the actual comparison targets, excluding command words
    vs_pattern = r'(?:compare|difference\s+between)?\s*([A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*)*)\s+(?:vs|versus)\s+([A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*)*?)(?:\s+(?:for|in|when|with|regarding|on)|$)'
    vs_matches = re.findall(vs_pattern, query, re.IGNORECASE)
    for match in vs_matches:
        for entity in match:
            entity = entity.strip()
            if entity and entity.lower() not in ['compare', 'difference'] and entity not in entities:
                entities.append(entity)
    
    # Extract capitalized phrases (simple heuristic)
    # Look for sequences of capitalized words
    capitalized_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
    capitalized_matches = re.findall(capitalized_pattern, query)
    
    # Filter out common words at start of sentences and question words
    common_start_words = {
        'The', 'A', 'An', 'This', 'That', 'What', 'How', 'Why', 'Who', 
        'When', 'Where', 'Which', 'Whose', 'Compare', 'Explain', 'Describe',
        'List', 'Name', 'Identify', 'Define', 'Analyze', 'Evaluate'
    }
    # Also filter out phrases containing prepositions that indicate clauses
    prepositions = {'for', 'in', 'when', 'with', 'regarding', 'on', 'after', 'before', 'during'}
    
    for match in capitalized_matches:
        words = match.split()
        # Filter out phrases that start with common words
        if words[0] in common_start_words:
            continue
        # Filter out phrases containing prepositions (likely clauses)
        if any(word.lower() in prepositions for word in words):
            continue
        # Filter out single common words, but keep multi-word phrases
        if len(words) > 1 or (len(words) == 1 and words[0] not in common_start_words):
            entities.append(match)
    
    # Extract technical terms (words with underscores, camelCase, etc.)
    tech_pattern = r'\b[a-z]+_[a-z]+\b|[A-Z][a-z]+[A-Z][a-z]+\b'
    entities.extend(re.findall(tech_pattern, query))
    
    # Extract single-letter uppercase entities (X, Y, Z, etc.)
    # But exclude common single-letter words
    single_letter_pattern = r'\b[A-Z]\b'
    single_letter_matches = re.findall(single_letter_pattern, query)
    common_single_letters = {'A', 'I'}
    for match in single_letter_matches:
        if match not in common_start_words and match not in common_single_letters:
            # Only add if it's not part of a larger word
            # Check if it's surrounded by spaces or punctuation
            if re.search(rf'(?<!\w){match}(?!\w)', query):
                entities.append(match)
    
    # Extract all-uppercase acronyms (2+ letters)
    acronym_pattern = r'\b[A-Z]{2,}\b'
    acronym_matches = re.findall(acronym_pattern, query)
    for match in acronym_matches:
        if match not in entities:
            entities.append(match)
    
    # Extract mixed-case alphanumeric terms (like BM25, GPT2, etc.)
    mixed_pattern = r'\b[A-Z]{2,}[0-9]+\b'
    mixed_matches = re.findall(mixed_pattern, query)
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


def extract_keywords(query: str) -> List[str]:
    """
    Extract important keywords from the query.
    
    This function removes stop words and extracts content-bearing terms.
    
    Args:
        query: The user query string
        
    Returns:
        List of keyword strings
    """
    # Common stop words to filter
    stop_words = {
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
        'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their', 'this',
        'that', 'these', 'those', 'what', 'which', 'who', 'whom', 'whose',
        'where', 'when', 'why', 'how', 'and', 'or', 'but', 'if', 'because',
        'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
        'against', 'between', 'into', 'through', 'during', 'before', 'after',
        'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
        'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there',
        'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
        'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
        'too', 'very', 'just', 'also', 'now', 'please', 'tell', 'give', 'show',
    }
    
    # Tokenize and filter
    words = re.findall(r'\b\w+\b', query.lower())
    keywords = [word for word in words if word not in stop_words and len(word) > 2]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_keywords = []
    for keyword in keywords:
        if keyword not in seen:
            seen.add(keyword)
            unique_keywords.append(keyword)
    
    return unique_keywords


def extract_temporal_markers(query: str) -> List[str]:
    """
    Extract temporal markers from the query.
    
    Args:
        query: The user query string
        
    Returns:
        List of temporal marker strings
    """
    temporal_markers = []
    
    # Time-related patterns
    temporal_patterns = [
        r'\b\d{4}\b',  # Years
        r'\b\d{1,2}(?:st|nd|rd|th)\s+(?:century|decade|millennium)\b',  # Centuries, etc.
        r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b',
        r'\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
        r'\b(?:morning|afternoon|evening|night|noon|midnight)\b',
        r'\b(?:yesterday|today|tomorrow)\b',
        r'\b(?:past|present|future|history)\b',
        r'\b(?:ancient|modern|contemporary)\b',
        r'\b(?:before|after|during|prior|subsequent)\b',
        r'\b(?:first|second|third|last|final)\b',
        r'\b(?:early|late)\b',
    ]
    
    query_lower = query.lower()
    for pattern in temporal_patterns:
        matches = re.findall(pattern, query_lower)
        temporal_markers.extend(matches)
    
    # Remove duplicates
    return list(set(temporal_markers))


def extract_time_words(query: str) -> List[str]:
    """
    Extract time words from the query for evidence planning.
    
    This function extracts temporal markers that indicate sequence,
    chronology, or temporal relationships in the query.
    
    Args:
        query: The user query string
        
    Returns:
        List of time word strings
    """
    time_words = []
    
    # Temporal words
    temporal_word_list = [
        'before', 'after', 'during', 'later', 'earlier', 'first', 'next',
        'then', 'eventually', 'timeline', 'sequence', 'chronology', 'when',
        'while', 'until', 'since'
    ]
    
    query_lower = query.lower()
    for word in temporal_word_list:
        if word in query_lower:
            time_words.append(word)
    
    # Also extract from temporal_markers for completeness
    temporal_markers = extract_temporal_markers(query)
    time_words.extend(temporal_markers)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_time_words = []
    for word in time_words:
        if word not in seen:
            seen.add(word)
            unique_time_words.append(word)
    
    return unique_time_words


def extract_comparison_targets(query: str) -> List[str]:
    """
    Extract entities being compared in comparative queries.
    
    Args:
        query: The user query string
        
    Returns:
        List of comparison target strings
    """
    q = query.strip().rstrip(".?!")
    
    patterns = [
        r"compare\s+(.+?)\s+and\s+(.+)$",
        r"compare\s+(.+?)\s+vs\.?\s+(.+)$",
        r"difference\s+between\s+(.+?)\s+and\s+(.+)$",
        r"(.+?)\s+vs\.?\s+(.+)$",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, q, flags=re.IGNORECASE)
        if match:
            left = match.group(1).strip()
            right = match.group(2).strip()
            return [left, right]
    
    return []


def extract_question_focus(query: str, query_type: str) -> Optional[str]:
    """
    Extract the main focus or constraint of the question.
    
    This function identifies what the question is specifically asking about,
    removing question words and extracting the core subject matter.
    
    Args:
        query: The user query string
        query_type: The classified query type
        
    Returns:
        The question focus string, or None if not detected
    """
    query_lower = query.lower()
    
    # Remove question words at the start
    question_starters = [
        'what', 'who', 'when', 'where', 'why', 'how', 'which', 'whose',
        'is', 'are', 'was', 'were', 'do', 'does', 'did', 'can', 'could',
        'would', 'should', 'will', 'may', 'might'
    ]
    
    focus = query_lower
    
    # Remove question words at start
    for starter in question_starters:
        if focus.startswith(starter):
            focus = focus[len(starter):].strip()
            # Remove auxiliary verbs
            if focus.startswith(('is ', 'are ', 'was ', 'were ', 'do ', 'does ', 'did ')):
                focus = focus.split(maxsplit=1)[1] if len(focus.split()) > 1 else focus
            break
    
    # For comparative queries, extract the constraint after comparison markers
    if query_type == 'comparative':
        # Pattern: "Compare A vs B for X" or "Compare A and B for X"
        for_marker = r'(?:for|in|when|with|regarding)\s+(.+)$'
        for_match = re.search(for_marker, focus)
        if for_match:
            return for_match.group(1).strip()
    
    # For temporal queries, extract what happened
    if query_type == 'temporal':
        # Remove temporal markers to get the core event
        temporal_words = ['before', 'after', 'during', 'when', 'while', 'until', 'since']
        for word in temporal_words:
            focus = focus.replace(word, '')
        return focus.strip()
    
    # For procedural queries, extract the action
    if query_type == 'procedural':
        # Pattern: "How to X" or "How do I X"
        if focus.startswith('to '):
            focus = focus[3:]
        return focus.strip()
    
    # For analytical questions, extract the subject of analysis
    if query_type == 'analytical':
        # Pattern: "What does X mean" or "What does X symbolize"
        if ' mean' in focus:
            focus = focus.split(' mean')[0]
        elif ' symbolize' in focus:
            focus = focus.split(' symbolize')[0]
        elif ' represent' in focus:
            focus = focus.split(' represent')[0]
        return focus.strip()
    
    # Default: return the cleaned focus
    if focus and len(focus) > 2:
        return focus.strip()
    
    return None


def detect_ambiguity(query: str) -> List[str]:
    """
    Detect ambiguity signals in the query.
    
    Args:
        query: The user query string
        
    Returns:
        List of ambiguity flag strings
    """
    ambiguity_flags = []
    
    # Check for pronouns without clear antecedents
    pronoun_patterns = [
        r'\bit\b',
        r'\bthis\b',
        r'\bthat\b',
        r'\bthey\b',
        r'\bthem\b',
        r'\bhe\b',
        r'\bshe\b',
    ]
    
    query_lower = query.lower()
    for pattern in pronoun_patterns:
        if re.search(pattern, query_lower):
            # Check if pronoun appears at start or without clear noun nearby
            clean_pattern = pattern.strip(r'\\b')
            ambiguity_flags.append(f"pronoun: {clean_pattern}")
    
    # Check for vague references
    vague_patterns = [
        r'\bthe thing\b',
        r'\bthe one\b',
        r'\bsomething\b',
        r'\bsomeone\b',
        r'\bthere\b',
        r'\bhere\b',
    ]
    
    for pattern in vague_patterns:
        if re.search(pattern, query_lower):
            clean_pattern = pattern.strip(r'\\b')
            ambiguity_flags.append(f"vague_reference: {clean_pattern}")
    
    # Check for missing subjects (very short queries)
    if len(query.split()) < 3:
        ambiguity_flags.append("missing_context")
    
    # Check for "or" without clear alternatives
    if re.search(r'\bor\b', query_lower) and not re.search(r'\beither\b', query_lower):
        ambiguity_flags.append("ambiguous_alternative")
    
    return ambiguity_flags


__all__ = [
    'extract_entities',
    'extract_keywords',
    'extract_temporal_markers',
    'extract_time_words',
    'extract_comparison_targets',
    'extract_question_focus',
    'detect_ambiguity',
]
