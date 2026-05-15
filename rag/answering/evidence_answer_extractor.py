"""
Deterministic Factual Answer Extraction Module

This module provides deterministic extraction of factual answers from retrieved evidence
before invoking the LLM generator. This ensures:
- Grounded answers when evidence is confident
- Reduced hallucination risk
- Faster response times for factual queries
- Clear attribution to source evidence

Architecture:
    Query → Retrieval → Slot Coverage Validation → Evidence Answer Extraction →
    If successful: return deterministic answer
    Else: invoke generator

The system is:
- Generic: Works across domains without customization
- Grounded: Only extracts what is explicitly stated in evidence
- Conservative: Returns None for missing slots rather than hallucinating
- Explainable: Provides extraction metadata for debugging
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple

from ..core.config import EXTRACTION_CONFIG


@dataclass
class ExtractionResult:
    """
    Result of deterministic answer extraction from evidence.
    
    Attributes:
        success: Whether extraction was successful for all requested slots
        answer: The extracted answer string (if successful)
        extracted_slots: Dict mapping slot names to extracted values
        missing_slots: List of slots that could not be extracted
        confidence: Confidence score for the extraction (0.0 to 1.0)
        metadata: Additional extraction metadata for debugging
    """
    success: bool
    answer: Optional[str]
    extracted_slots: Dict[str, Optional[str]]
    missing_slots: List[str]
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


def extract_factual_answer_from_evidence(
    query: str,
    evidence_chunks: List[str],
    answer_targets: Optional[List[Any]] = None,
    expected_answer_slots: Optional[List[str]] = None,
    anchor_entity: Optional[str] = None,
) -> ExtractionResult:
    """
    Extract deterministic factual answers from retrieved evidence.
    
    This function attempts to extract direct answers from evidence chunks
    using generic patterns and entity-aware heuristics. If confident extraction
    succeeds for all requested slots, it returns a grounded answer without
    requiring LLM generation.
    
    Args:
        query: The original user query
        evidence_chunks: List of retrieved text chunks
        answer_targets: Optional list of AnswerTarget objects for semantic extraction
        expected_answer_slots: Optional list of slot names for backward compatibility
        anchor_entity: Optional anchor entity for possessive queries
        
    Returns:
        ExtractionResult with success status, answer, and metadata
        
    Example:
        Query: "What are the names of Entity's relation_a and relation_b?"
        Evidence: ["NAME, Entity’s relation_a", "relation_b. This is NAME."]
        
        Returns: ExtractionResult(
            success=True,
            answer="Entity’s relation_a is Name, and Entity’s relation_b is Name.",
            extracted_slots={"relation_a": "Name", "relation_b": "Name"},
            missing_slots=[],
            confidence=0.9
        )
    """
    # Normalize inputs
    if answer_targets:
        # Use semantic AnswerTarget objects
        slots = [target.target_type for target in answer_targets]
        anchor = answer_targets[0].anchor_entity if answer_targets else anchor_entity
        requested_attr = answer_targets[0].requested_attribute if answer_targets else None
    elif expected_answer_slots:
        # Backward compatibility: use expected_answer_slots
        slots = expected_answer_slots
        anchor = anchor_entity
        requested_attr = None
    else:
        # No slot information, cannot extract
        return ExtractionResult(
            success=False,
            answer=None,
            extracted_slots={},
            missing_slots=[],
            confidence=0.0,
            metadata={"reason": "no_slot_information"}
        )
    
    # Process chunks individually to avoid cross-matching
    extracted_slots = {}
    missing_slots = []
    extraction_confidences = {}
    
    for slot in slots:
        slot_lower = slot.lower()
        value = None
        confidence = 0.0
        
        # Try extraction patterns on each chunk individually
        for chunk in evidence_chunks:
            # Handle both string chunks and chunk objects
            chunk_text = chunk if isinstance(chunk, str) else getattr(chunk, 'text', str(chunk))
            
            chunk_value, chunk_confidence = _extract_slot_value(
                slot=slot_lower,
                evidence=chunk_text,
                anchor_entity=anchor,
                requested_attribute=requested_attr
            )
            
            # Use the highest confidence match across all chunks
            if chunk_value is not None and chunk_confidence > confidence:
                value = chunk_value
                confidence = chunk_confidence
        
        if value is not None and confidence >= EXTRACTION_CONFIG.min_confidence:
            extracted_slots[slot] = value
            extraction_confidences[slot] = confidence
        else:
            extracted_slots[slot] = None
            missing_slots.append(slot)
            extraction_confidences[slot] = 0.0
    
    # Determine overall success
    all_slots_found = len(missing_slots) == 0
    overall_confidence = (
        sum(extraction_confidences.values()) / len(extraction_confidences)
        if extraction_confidences else 0.0
    )
    
    # Build answer if successful
    answer = None
    if all_slots_found:
        answer = _build_answer_from_slots(
            extracted_slots=extracted_slots,
            anchor_entity=anchor,
            requested_attribute=requested_attr,
            query=query
        )
    elif extracted_slots:
        # Partial coverage: answer covered slots and mention missing
        answer = _build_partial_answer(
            extracted_slots=extracted_slots,
            missing_slots=missing_slots,
            anchor_entity=anchor
        )
    
    # Build metadata
    metadata = {
        "extraction_method": "deterministic_patterns",
        "total_slots": len(slots),
        "covered_slots": len(extracted_slots) - len(missing_slots),
        "slot_confidences": extraction_confidences,
        "anchor_entity": anchor,
        "requested_attribute": requested_attr,
    }
    
    return ExtractionResult(
        success=all_slots_found,
        answer=answer,
        extracted_slots=extracted_slots,
        missing_slots=missing_slots,
        confidence=overall_confidence,
        metadata=metadata
    )


def _extract_slot_value(
    slot: str,
    evidence: str,
    anchor_entity: Optional[str] = None,
    requested_attribute: Optional[str] = None,
) -> Tuple[Optional[str], float]:
    """
    Extract a single slot value from evidence using generic patterns.
    
    Args:
        slot: The slot name to extract (e.g., "son", "daughter")
        evidence: Combined evidence text
        anchor_entity: Optional anchor entity for context
        requested_attribute: Optional requested attribute (e.g., "name", "age")
        
    Returns:
        Tuple of (extracted_value, confidence_score)
    """
    evidence_lower = evidence.lower()
    
    # Escape anchor entity for regex safety
    anchor_pattern = re.escape(anchor_entity) if anchor_entity else None
    
    def validate_extraction(candidate: str) -> bool:
        """Validate that extraction is not a configured blocked extraction."""
        if not candidate:
            return False
        candidate_lower = candidate.lower().strip()
        return candidate_lower not in EXTRACTION_CONFIG.bad_extractions
    
    # Pattern 1: "NAME, anchor_entity's ... slot" (STRONG PATTERN - all-caps or mixed case)
    # Example pattern: "NAME, Entity’s relation" -> relation = Name
    # Example pattern: "NAME, Entity’s relation" -> relation = Name (curly apostrophe)
    if anchor_pattern:
        pattern1 = rf"\b([A-Z][A-Z]+|[A-Z][a-z]+),\s+{anchor_pattern}[’']s[^.]*?\b{slot}\b"
        match = re.search(pattern1, evidence, re.IGNORECASE)
        if match:
            candidate = match.group(1).title()
            if validate_extraction(candidate):
                return candidate, 0.9
    
    # Pattern 2: "slot. This is NAME" (SCREENPLAY INTRODUCTION - all-caps or mixed case)
    # Example pattern: "relation. This is NAME" -> relation = Name
    pattern2 = rf"\b{slot}\b[^.]*?\.\s+This is\s+([A-Z][A-Z]+|[A-Z][a-z]+)"
    match = re.search(pattern2, evidence, re.IGNORECASE)
    if match:
        candidate = match.group(1).title()
        if validate_extraction(candidate):
            return candidate, 0.85
    
    # Pattern 3: "slot. Named NAME" or "slot. Called NAME" (all-caps or mixed case)
    # Example pattern: "relation. Named NAME" -> relation = Name
    pattern3 = rf"\b{slot}\b[^.]*?\.\s+(?:Named|Called|named|called)\s+([A-Z][A-Z]+|[A-Z][a-z]+)"
    match = re.search(pattern3, evidence, re.IGNORECASE)
    if match:
        candidate = match.group(1).title()
        if validate_extraction(candidate):
            return candidate, 0.85
    
    # Pattern 4: "anchor_entity's slot is NAME" (supports both apostrophe types)
    if anchor_pattern:
        pattern4 = rf"{anchor_pattern}[’']s\s+\b{slot}\b\s+(?:is|was)\s+([A-Z][A-Z]+|[A-Z][a-z]+)"
        match = re.search(pattern4, evidence, re.IGNORECASE)
        if match:
            candidate = match.group(1).title()
            if validate_extraction(candidate):
                return candidate, 0.85
    
    # Pattern 5: "NAME is anchor_entity's slot" (supports both apostrophe types)
    if anchor_pattern:
        pattern5 = rf"([A-Z][A-Z]+|[A-Z][a-z]+)\s+(?:is|was)\s+{anchor_pattern}[’']s\s+\b{slot}\b"
        match = re.search(pattern5, evidence, re.IGNORECASE)
        if match:
            candidate = match.group(1).title()
            if validate_extraction(candidate):
                return candidate, 0.85
    
    # Pattern 6: "slot: NAME" or "slot is NAME" (weaker, but validated)
    pattern6 = rf"\b{slot}\b\s*(?:is|:)\s+([A-Z][A-Z]+|[A-Z][a-z]+)"
    match = re.search(pattern6, evidence, re.IGNORECASE)
    if match:
        candidate = match.group(1).title()
        if validate_extraction(candidate):
            return candidate, 0.7
    
    # Pattern 7: "The slot of anchor_entity is NAME"
    if anchor_pattern:
        pattern7 = rf"The\s+\b{slot}\b\s+of\s+{anchor_pattern}\s+(?:is|was)\s+([A-Z][A-Z]+|[A-Z][a-z]+)"
        match = re.search(pattern7, evidence, re.IGNORECASE)
        if match:
            candidate = match.group(1).title()
            if validate_extraction(candidate):
                return candidate, 0.7
    
    # Pattern 8: Generic: "anchor_entity slot NAME" (weaker, but validated)
    if anchor_pattern:
        pattern8 = rf"{anchor_pattern}\s+\b{slot}\b\s+([A-Z][A-Z]+|[A-Z][a-z]+)"
        match = re.search(pattern8, evidence, re.IGNORECASE)
        if match:
            candidate = match.group(1).title()
            if validate_extraction(candidate):
                return candidate, 0.6
    
    # No confident match found
    return None, 0.0


def _build_answer_from_slots(
    extracted_slots: Dict[str, Optional[str]],
    anchor_entity: Optional[str],
    requested_attribute: Optional[str],
    query: str,
) -> str:
    """
    Build a natural language answer from extracted slot values.
    
    Args:
        extracted_slots: Dict mapping slot names to extracted values
        anchor_entity: Optional anchor entity
        requested_attribute: Optional requested attribute
        query: Original query for context
        
    Returns:
        Natural language answer string
    """
    if not extracted_slots:
        return "I could not extract the answer from the provided evidence."
    
    # Determine answer format based on requested attribute
    if requested_attribute == 'name':
        prefix = "named"
    elif requested_attribute == 'age':
        prefix = "aged"
    elif requested_attribute == 'location':
        prefix = "located"
    elif requested_attribute == 'role':
        prefix = "with the role"
    else:
        prefix = ""
    
    # Build answer with anchor entity
    if anchor_entity:
        if len(extracted_slots) == 1:
            slot, value = list(extracted_slots.items())[0]
            if prefix:
                return f"{anchor_entity}'s {slot} is {prefix} {value}."
            else:
                return f"{anchor_entity}'s {slot} is {value}."
        else:
            # Multiple slots
            slot_parts = []
            for slot, value in extracted_slots.items():
                if prefix:
                    slot_parts.append(f"{anchor_entity}'s {slot} is {prefix} {value}")
                else:
                    slot_parts.append(f"{anchor_entity}'s {slot} is {value}")
            
            if len(slot_parts) == 2:
                return f"{slot_parts[0]}, and {slot_parts[1]}."
            else:
                return ", ".join(slot_parts[:-1]) + f", and {slot_parts[-1]}."
    else:
        # No anchor entity
        if len(extracted_slots) == 1:
            slot, value = list(extracted_slots.items())[0]
            if prefix:
                return f"The {slot} is {prefix} {value}."
            else:
                return f"The {slot} is {value}."
        else:
            # Multiple slots without anchor
            slot_parts = []
            for slot, value in extracted_slots.items():
                if prefix:
                    slot_parts.append(f"The {slot} is {prefix} {value}")
                else:
                    slot_parts.append(f"The {slot} is {value}")
            
            if len(slot_parts) == 2:
                return f"{slot_parts[0]}, and {slot_parts[1]}."
            else:
                return ", ".join(slot_parts[:-1]) + f", and {slot_parts[-1]}."


def _build_partial_answer(
    extracted_slots: Dict[str, Optional[str]],
    missing_slots: List[str],
    anchor_entity: Optional[str],
) -> str:
    """
    Build a partial answer when some slots are missing.
    
    Args:
        extracted_slots: Dict mapping slot names to extracted values
        missing_slots: List of slots that could not be extracted
        anchor_entity: Optional anchor entity
        
    Returns:
        Partial answer string with explicit mention of missing slots
    """
    # Build the covered part
    covered_parts = []
    for slot, value in extracted_slots.items():
        if value:
            if anchor_entity:
                covered_parts.append(f"{anchor_entity}'s {slot} is {value}")
            else:
                covered_parts.append(f"The {slot} is {value}")
    
    if not covered_parts:
        return f"I could not extract the answer from the provided evidence."
    
    # Build answer
    if len(covered_parts) == 1:
        answer = covered_parts[0]
    else:
        answer = ", ".join(covered_parts[:-1]) + f", and {covered_parts[-1]}."
    
    # Add missing slots
    if missing_slots:
        missing_str = ", ".join(missing_slots)
        answer += f" However, I could not find information about {missing_str} in the evidence."
    
    return answer


def should_use_deterministic_extraction(
    query_analysis: Any,
) -> bool:
    """
    Determine whether to use deterministic extraction based on query analysis.
    
    Args:
        query_analysis: QueryAnalysis object from query understanding
        
    Returns:
        True if deterministic extraction should be attempted
    """
    # Use deterministic extraction for:
    # 1. Factual queries with entity coverage
    # 2. High confidence classification
    # 3. Explicit answer targets
    
    if not query_analysis:
        return False
    
    # Only attempt for factual queries with entity coverage
    if query_analysis.query_type != "factual":
        return False
    
    # Require entity coverage detection
    if not query_analysis.requires_entity_coverage:
        return False
    
    # Require high confidence in classification
    if query_analysis.confidence < 0.7:
        return False
    
    # Require answer targets or expected slots
    if not query_analysis.answer_targets and not query_analysis.expected_answer_slots:
        return False
    
    return True


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    'ExtractionResult',
    'extract_factual_answer_from_evidence',
    'should_use_deterministic_extraction',
]
