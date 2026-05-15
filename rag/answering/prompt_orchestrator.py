"""
Prompt Orchestration Layer for Adaptive RAG System - Phase 7

This module provides a query-aware prompt orchestration system that dynamically
changes prompt structure, reasoning style, grounding behavior, and answer formatting
based on query semantics.

Architecture:
    Query → QueryAnalysis → Answer Style Selection → Prompt Template →
    Dynamic Enrichment → Final Prompt → Generator

The system is:
- Query-aware: Uses QueryAnalysis metadata to adapt prompts
- Modular: Each answer style has its own template and behavior
- Extensible: Easy to add new answer styles and prompt strategies
- Generator-agnostic: Not tied to any specific model backend
- Safety-focused: Minimizes hallucination with strong grounding constraints
- Traceable: Rich metadata for analytics and debugging
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass


# ============================================================================
# PROMPT TEMPLATE DEFINITIONS
# ============================================================================

PROMPT_TEMPLATES = {
    "short_direct": {
        "system_instruction": "Answer using ONLY the provided context.",
        "behavior_instruction": "Give the shortest correct answer. Do not explain unless necessary.",
        "grounding_instruction": "Extractive preference - prioritize direct extraction from context.",
        "formatting_guidance": "Short, direct answer only.",
        "hallucination_control": "If answer is missing say: 'I don't know from the provided context.'",
        "verbosity_control": "Minimal - single sentence or phrase preferred.",
    },
    "brief_explanation": {
        "system_instruction": "Answer using ONLY the provided context.",
        "behavior_instruction": "Explain briefly in 2-4 sentences with grounded reasoning.",
        "grounding_instruction": "Provide a concise explanation grounded in the context.",
        "formatting_guidance": "2-4 sentence reasoning, concise and clear.",
        "hallucination_control": "If answer is missing say: 'I don't know from the provided context.'",
        "verbosity_control": "Moderate - brief but complete reasoning.",
    },
    "comparison": {
        "system_instruction": "Answer using ONLY the provided context.",
        "behavior_instruction": "Provide a balanced comparison of the entities/systems/methods mentioned.",
        "grounding_instruction": "Compare both entities explicitly. Preserve neutrality. Avoid entity bias.",
        "formatting_guidance": "Organize clearly with similarities, differences, and best use cases.",
        "hallucination_control": "If answer is missing say: 'I don't know from the provided context.'",
        "verbosity_control": "Structured comparison - use sections or bullet points.",
    },
    "timeline": {
        "system_instruction": "Answer using ONLY the provided context.",
        "behavior_instruction": "Provide a chronological timeline of events in order.",
        "grounding_instruction": "Preserve event order. Emphasize sequence. Be aware of before/after relationships.",
        "formatting_guidance": "Chronological explanation with clear sequence indicators.",
        "hallucination_control": "If answer is missing say: 'I don't know from the provided context.'",
        "verbosity_control": "Sequential - emphasize temporal ordering.",
    },
    "steps": {
        "system_instruction": "Answer using ONLY the provided context.",
        "behavior_instruction": "Provide step-by-step instructions or process description.",
        "grounding_instruction": "Preserve process ordering. Use numbered steps for clarity.",
        "formatting_guidance": "Numbered steps with instructional reasoning.",
        "hallucination_control": "If answer is missing say: 'I don't know from the provided context.'",
        "verbosity_control": "Sequential - ordered steps with clear structure.",
    },
    "interpretation": {
        "system_instruction": "Answer using ONLY the provided context.",
        "behavior_instruction": "Provide an analytical interpretation of the meaning, significance, or implications.",
        "grounding_instruction": "Interpretive explanation grounded in context. Be implication-aware.",
        "formatting_guidance": "Analytical explanation focusing on meaning and significance.",
        "hallucination_control": "If answer is missing say: 'I don't know from the provided context.'",
        "verbosity_control": "Analytical - depth appropriate to context evidence.",
    },
    "synthesized": {
        "system_instruction": "Answer using ONLY the provided context.",
        "behavior_instruction": "Synthesize information from multiple sources to provide a comprehensive answer.",
        "grounding_instruction": "Combine evidence across chunks. Multi-step reasoning. Connect different facts.",
        "formatting_guidance": "Connected explanations that synthesize multiple pieces of evidence.",
        "hallucination_control": "If answer is missing say: 'I don't know from the provided context.'",
        "verbosity_control": "Comprehensive - multi-hop reasoning with evidence synthesis.",
    },
    "clarify_or_refuse": {
        "system_instruction": "Answer using ONLY the provided context.",
        "behavior_instruction": "If the question is vague or lacks context, ask for clarification. If the context doesn't contain the answer, state that clearly.",
        "grounding_instruction": "Prioritize truthfulness over helpfulness. Avoid fabrication. Explain ambiguity.",
        "formatting_guidance": "Clarification request or clear refusal with explanation.",
        "hallucination_control": "Explicitly state uncertainty. Do not fabricate answers.",
        "verbosity_control": "Concise - focus on clarification or refusal.",
    },
    "default": {
        "system_instruction": "Answer using ONLY the provided context.",
        "behavior_instruction": "Provide a clear and helpful answer based on the context.",
        "grounding_instruction": "Ground your answer in the provided context.",
        "formatting_guidance": "Clear, natural language response.",
        "hallucination_control": "If answer is missing say: 'I don't know from the provided context.'",
        "verbosity_control": "Balanced - appropriate to the query and context.",
    },
}


# ============================================================================
# DYNAMIC ENRICHMENT FUNCTIONS
# ============================================================================

def enrich_for_comparison(
    prompt: str,
    query_analysis: Optional[Any],
) -> str:
    """
    Enrich prompt for comparative queries by injecting comparison targets.
    
    Args:
        prompt: Base prompt string
        query_analysis: QueryAnalysis object with metadata
        
    Returns:
        Enriched prompt string
    """
    if query_analysis is None:
        return prompt
    
    comparison_targets = getattr(query_analysis, 'comparison_targets', [])
    if comparison_targets:
        targets_str = " and ".join(comparison_targets)
        enrichment = f"\n\nComparison Focus: {targets_str}"
        prompt = prompt.replace("[/INST]", enrichment + "[/INST]")
    
    return prompt


def enrich_for_temporal(
    prompt: str,
    query_analysis: Optional[Any],
) -> str:
    """
    Enrich prompt for temporal queries by emphasizing chronology.
    
    Args:
        prompt: Base prompt string
        query_analysis: QueryAnalysis object with metadata
        
    Returns:
        Enriched prompt string
    """
    if query_analysis is None:
        return prompt
    
    time_words = getattr(query_analysis, 'time_words', [])
    if time_words:
        enrichment = "\n\nTemporal Focus: Emphasize chronological order and sequence."
        prompt = prompt.replace("[/INST]", enrichment + "[/INST]")
    
    return prompt


def enrich_for_multi_hop(
    prompt: str,
    query_analysis: Optional[Any],
) -> str:
    """
    Enrich prompt for multi-hop queries by emphasizing evidence connection.
    
    Args:
        prompt: Base prompt string
        query_analysis: QueryAnalysis object with metadata
        
    Returns:
        Enriched prompt string
    """
    if query_analysis is None:
        return prompt
    
    if getattr(query_analysis, 'requires_multi_hop', False):
        enrichment = "\n\nMulti-hop Reasoning: Connect evidence from different parts of the context to answer."
        prompt = prompt.replace("[/INST]", enrichment + "[/INST]")
    
    return prompt


def enrich_for_unanswerable(
    prompt: str,
    query_analysis: Optional[Any],
) -> str:
    """
    Enrich prompt for unanswerable queries by emphasizing refusal safety.
    
    Args:
        prompt: Base prompt string
        query_analysis: QueryAnalysis object with metadata
        
    Returns:
        Enriched prompt string
    """
    if query_analysis is None:
        return prompt
    
    ambiguity_flags = getattr(query_analysis, 'ambiguity_flags', [])
    if ambiguity_flags:
        enrichment = "\n\nSafety Priority: If the question is ambiguous or the answer is not in the context, explicitly state this rather than guessing."
        prompt = prompt.replace("[/INST]", enrichment + "[/INST]")
    
    return prompt


def enrich_for_entity_coverage(
    prompt: str,
    query_analysis: Optional[Any],
    slot_coverage_info: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Enrich prompt for entity coverage queries by handling partial slot coverage.
    
    This function adds instructions to the prompt about how to handle missing
    answer slots - the generator should answer covered parts and clearly state
    what information is missing from the context.
    
    Args:
        prompt: Base prompt string
        query_analysis: QueryAnalysis object with metadata
        slot_coverage_info: Dictionary with slot coverage information from validation
        
    Returns:
        Enriched prompt string
    """
    if query_analysis is None or slot_coverage_info is None:
        return prompt
    
    requires_entity_coverage = getattr(query_analysis, 'requires_entity_coverage', False)
    if not requires_entity_coverage:
        return prompt
    
    missing_slots = slot_coverage_info.get('missing_slots', [])
    if missing_slots:
        # If some slots are missing, instruct the generator to be explicit about what's missing
        missing_slots_str = ", ".join(missing_slots)
        enrichment = f"\n\nPartial Coverage Warning: The context may not contain information about: {missing_slots_str}. Answer the parts you can find in the context, and clearly state what information is missing. Do not guess or fabricate missing information."
        prompt = prompt.replace("[/INST]", enrichment + "[/INST]")
    else:
        # All slots are covered, reinforce completeness
        enrichment = "\n\nMulti-slot Answer: Answer all requested parts of the question based on the provided context."
        prompt = prompt.replace("[/INST]", enrichment + "[/INST]")
    
    return prompt


# ============================================================================
# PROMPT BUILDING FUNCTIONS
# ============================================================================

def build_prompt_from_template(
    template: Dict[str, str],
    query: str,
    context: str,
) -> str:
    """
    Build a prompt from a template dictionary.
    
    Args:
        template: Template dictionary with prompt components
        query: User question
        context: Retrieved context text
        
    Returns:
        Formatted prompt string
    """
    system_instruction = template["system_instruction"]
    behavior_instruction = template["behavior_instruction"]
    grounding_instruction = template["grounding_instruction"]
    formatting_guidance = template["formatting_guidance"]
    hallucination_control = template["hallucination_control"]
    
    prompt = f"""<s>[INST]
{system_instruction}

{behavior_instruction}

{grounding_instruction}

{formatting_guidance}

{hallucination_control}

Question:
{query}

Context:
{context}
[/INST]
"""
    return prompt


def get_answer_style_from_query(
    query_analysis: Optional[Any],
    fallback_style: str = "default",
) -> str:
    """
    Determine the appropriate answer style from query analysis.
    
    Args:
        query_analysis: QueryAnalysis object with metadata
        fallback_style: Default style if analysis is unavailable
        
    Returns:
        Answer style string
    """
    if query_analysis is None:
        return fallback_style
    
    # Use the answer_style from query analysis if available
    answer_style = getattr(query_analysis, 'answer_style', None)
    if answer_style and answer_style in PROMPT_TEMPLATES:
        return answer_style
    
    # Fallback to query type mapping
    query_type = getattr(query_analysis, 'query_type', None)
    if query_type:
        style_mapping = {
            "factual": "short_direct",
            "reasoning": "brief_explanation",
            "comparative": "comparison",
            "temporal": "timeline",
            "procedural": "steps",
            "analytical": "interpretation",
            "multi_hop": "synthesized",
            "unanswerable": "clarify_or_refuse",
        }
        return style_mapping.get(query_type, fallback_style)
    
    return fallback_style


def build_prompt(
    query: str,
    context: str,
    answer_style: str = "brief_explanation",
    query_analysis: Optional[Any] = None,
    slot_coverage_info: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build a query-aware prompt for generation.
    
    This function orchestrates prompt generation based on query semantics
    and answer style requirements. It uses modular templates and dynamic
    enrichment to create contextually appropriate prompts.
    
    Args:
        query: User question
        context: Retrieved context text
        answer_style: Preferred answer style (short_direct, brief_explanation, 
                      comparison, timeline, steps, interpretation, synthesized,
                      clarify_or_refuse, default)
        query_analysis: Optional QueryAnalysis object with metadata for dynamic enrichment
        slot_coverage_info: Optional dictionary with slot coverage information for entity coverage queries
    
    Returns:
        Formatted prompt string
    
    Example:
        >>> analysis = understand_query("Compare A vs B")
        >>> prompt = build_prompt(
        ...     query="Compare A vs B",
        ...     context=raw_context,
        ...     answer_style=analysis.answer_style,
        ...     query_analysis=analysis,
        ... )
    """
    # Determine answer style if not explicitly provided
    if answer_style == "default" or answer_style not in PROMPT_TEMPLATES:
        answer_style = get_answer_style_from_query(query_analysis, fallback_style="default")
    
    # Get the template for the answer style
    template = PROMPT_TEMPLATES.get(answer_style, PROMPT_TEMPLATES["default"])
    
    # Build base prompt from template
    prompt = build_prompt_from_template(template, query, context)
    
    # Apply dynamic enrichments based on query analysis
    if query_analysis is not None:
        query_type = getattr(query_analysis, 'query_type', None)
        
        # Comparative enrichment
        if query_type == "comparative":
            prompt = enrich_for_comparison(prompt, query_analysis)
        
        # Temporal enrichment
        elif query_type == "temporal":
            prompt = enrich_for_temporal(prompt, query_analysis)
        
        # Multi-hop enrichment
        elif query_type == "multi_hop":
            prompt = enrich_for_multi_hop(prompt, query_analysis)
        
        # Unanswerable enrichment
        elif query_type == "unanswerable":
            prompt = enrich_for_unanswerable(prompt, query_analysis)
        
        # Phase 12-prep: Entity coverage enrichment
        # Apply when slot_coverage_info is provided
        if slot_coverage_info:
            prompt = enrich_for_entity_coverage(prompt, query_analysis, slot_coverage_info)
    
    return prompt


# ============================================================================
# PROMPT METADATA EXTRACTION
# ============================================================================

def extract_prompt_metadata(
    query_analysis: Optional[Any],
    answer_style: str,
) -> Dict[str, Any]:
    """
    Extract metadata about the prompt orchestration for traceability.
    
    Args:
        query_analysis: QueryAnalysis object with metadata
        answer_style: The answer style used for prompting
        
    Returns:
        Dictionary of prompt orchestration metadata
    """
    metadata = {
        "answer_style": answer_style,
    }
    
    if query_analysis is not None:
        metadata["query_type"] = getattr(query_analysis, 'query_type', None)
        metadata["confidence"] = getattr(query_analysis, 'confidence', None)
        metadata["entities"] = getattr(query_analysis, 'entities', [])
        metadata["keywords"] = getattr(query_analysis, 'keywords', [])
        metadata["ambiguity_flags"] = getattr(query_analysis, 'ambiguity_flags', [])
        metadata["requires_reasoning"] = getattr(query_analysis, 'requires_reasoning', False)
        metadata["requires_multi_hop"] = getattr(query_analysis, 'requires_multi_hop', False)
        metadata["requires_temporal_ordering"] = getattr(query_analysis, 'requires_temporal_ordering', False)
    
    return metadata


# ============================================================================
# LEGACY COMPATIBILITY
# ============================================================================

def build_legacy_prompt(
    query: str,
    context: str,
    query_type: Optional[str] = None,
    answer_style: Optional[str] = None,
) -> str:
    """
    Legacy prompt builder for backward compatibility.
    
    This function maintains the old interface while delegating to the new
    orchestration system. It is kept for backward compatibility but new code
    should use build_prompt() with QueryAnalysis.
    
    Args:
        query: User question
        context: Retrieved context text
        query_type: Legacy query type (factual, reasoning, etc.)
        answer_style: Legacy answer style
        
    Returns:
        Formatted prompt string
    """
    # Map legacy query types to answer styles
    if answer_style is None:
        if query_type == "factual":
            answer_style = "short_direct"
        else:
            answer_style = "brief_explanation"
    
    return build_prompt(query, context, answer_style=answer_style)
