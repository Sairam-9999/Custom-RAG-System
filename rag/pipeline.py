"""
RAG Pipeline Module

This module provides the main orchestration for the RAG pipeline,
including retrieval, slot-aware retrieval, and answer validation.
"""

import time
import re
from typing import Dict, Any, Optional, List

# Import from new modular structure
from .core.types import CompressedContext
from .core.config import EMBEDDING_CONFIG, CHUNKING_CONFIG, RUNTIME_CONFIG
from .indexing import chunk_text, get_embeddings, VectorStore, IndexCache
from .retrieval import retrieve
from .query import understand_query
from .answering.prompt_orchestrator import build_prompt, extract_prompt_metadata
from .answering.validator import validate_answer, revise_answer
from .answering.evidence_answer_extractor import (
    extract_factual_answer_from_evidence,
    should_use_deterministic_extraction,
    ExtractionResult,
)
from .retrieval import (
    decide_retry,
    expand_retrieval_policy,
    select_best_answer,
    RetryAttempt,
    should_continue_retry,
    build_retry_metadata,
)


# ============================================================================
# SLOT-AWARE RETRIEVAL
# ============================================================================

def build_slot_queries(query: str, expected_answer_slots: List[str], anchor_entity: Optional[str] = None) -> List[str]:
    """
    Build focused retrieval queries for each expected answer slot.
    
    This function creates targeted queries that focus on retrieving evidence
    for each specific answer slot in a multi-slot question.
    
    Args:
        query: The original user query
        expected_answer_slots: List of answer slots that need evidence coverage
        anchor_entity: The main entity/subject for the slots (e.g., "Cooper")
        
    Returns:
        List of focused queries, one per slot
    """
    slot_queries = []
    
    # If anchor_entity is provided, use it to build focused queries
    if anchor_entity:
        for slot in expected_answer_slots:
            # Build query: "anchor slot name" (e.g., "cooper son name", "cooper daughter name")
            slot_queries.append(f"{anchor_entity} {slot} name")
        return slot_queries
    
    # Fallback: extract from query if no anchor_entity provided
    query_lower = query.lower()
    
    # Extract the main subject/context from the query
    # For possessive patterns: "Cooper's son and daughter" -> "Cooper"
    possessive_match = re.search(r"(\w+)'s", query_lower)
    if possessive_match:
        main_subject = possessive_match.group(1)
        for slot in expected_answer_slots:
            slot_queries.append(f"{main_subject} {slot} name")
        return slot_queries
    
    # For "names of X and Y" patterns
    names_of_match = re.search(r"names?\s+of\s+(.+?)(?:\s+and\s+|\s|$)", query_lower, re.IGNORECASE)
    if names_of_match:
        main_subject = names_of_match.group(1)
        for slot in expected_answer_slots:
            slot_queries.append(f"{main_subject} {slot} name")
        return slot_queries
    
    # Fallback: use the slot itself as the query
    for slot in expected_answer_slots:
        slot_queries.append(f"{slot}")
    
    return slot_queries


def retrieve_for_slots(
    slot_queries: List[str],
    store,
    chunks,
    top_k_per_slot: int = 3,
) -> Dict[str, List]:
    """
    Retrieve evidence for each slot using focused queries.
    
    This function performs separate retrieval for each slot query
    to ensure each answer slot has dedicated supporting evidence.
    
    Args:
        slot_queries: List of focused queries for each slot
        store: Vector store for semantic retrieval
        chunks: List of document chunks
        top_k_per_slot: Number of chunks to retrieve per slot
        
    Returns:
        Dictionary mapping slot index to list of retrieval results
    """
    from .core.types import RetrievalResult
    
    slot_results = {}
    
    for i, slot_query in enumerate(slot_queries):
        results = retrieve(slot_query, store, chunks, top_k=top_k_per_slot)
        slot_results[i] = results
    
    return slot_results


def merge_and_deduplicate_chunks(
    slot_results: Dict[str, List],
    max_total_chunks: int = 10,
) -> List:
    """
    Merge and deduplicate chunks from slot-based retrieval.
    
    This function combines retrieval results from all slots,
    removes duplicate chunks, and preserves the highest-scoring versions.
    
    Args:
        slot_results: Dictionary mapping slot index to retrieval results
        max_total_chunks: Maximum number of chunks to return after merging
        
    Returns:
        List of unique, deduplicated retrieval results
    """
    from .core.types import RetrievalResult
    
    # Track seen chunks by text content
    seen_texts = set()
    merged_results = []
    
    # Collect all results from all slots
    all_results = []
    for slot_idx, results in slot_results.items():
        for result in results:
            all_results.append((result, slot_idx))
    
    # Sort by hybrid score (descending) to prioritize best chunks
    all_results.sort(key=lambda x: x[0].hybrid_score if hasattr(x[0], 'hybrid_score') else 0, reverse=True)
    
    # Deduplicate while preserving order
    for result, slot_idx in all_results:
        if result.text not in seen_texts:
            seen_texts.add(result.text)
            merged_results.append(result)
            
            if len(merged_results) >= max_total_chunks:
                break
    
    return merged_results


def validate_slot_coverage(
    expected_answer_slots: List[str],
    retrieved_chunks: List,
) -> Dict[str, Any]:
    """
    Validate whether each expected answer slot has supporting evidence.
    
    This function checks if the retrieved chunks contain evidence
    for each expected answer slot in the query.
    
    Args:
        expected_answer_slots: List of answer slots that need evidence
        retrieved_chunks: List of retrieved chunks
        
    Returns:
        Dictionary with coverage information:
        {
            "slot_coverage": Dict[str, bool],  # slot -> has_evidence
            "missing_slots": List[str],  # slots without evidence
            "coverage_ratio": float,  # percentage of slots covered
        }
    """
    slot_coverage = {}
    missing_slots = []
    
    # Normalize chunk text for searching
    chunk_texts_lower = " ".join([chunk.text.lower() for chunk in retrieved_chunks])
    
    for slot in expected_answer_slots:
        slot_lower = slot.lower()
        
        # Check if slot appears in any chunk
        # Use word boundary matching for more precise detection
        pattern = r'\b' + re.escape(slot_lower) + r'\b'
        if re.search(pattern, chunk_texts_lower):
            slot_coverage[slot] = True
        else:
            slot_coverage[slot] = False
            missing_slots.append(slot)
    
    coverage_ratio = len([s for s, covered in slot_coverage.items() if covered]) / len(expected_answer_slots) if expected_answer_slots else 1.0
    
    return {
        "slot_coverage": slot_coverage,
        "missing_slots": missing_slots,
        "coverage_ratio": coverage_ratio,
    }


def load_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def build_rag(file_path, workload_type="mixed", use_cache=True, cache_name=None):
    """
    Build RAG index with adaptive chunking based on expected workload.
    
    Args:
        file_path: Path to text file
        workload_type: Expected query workload type
            - "fact": Optimized for factual QA (small chunks, 220 chars, overlap=1)
            - "analytical": Optimized for reasoning/summaries (medium chunks, 450 chars, overlap=2)
            - "mixed": Default chunking (650 chars, overlap=3)
        use_cache: Whether to use cached index if available (default True)
        cache_name: Optional name for the cache (defaults to file hash)
    
    Returns:
        Tuple of (vector_store, chunks)
    """
    import time
    import hashlib
    from .indexing import IndexCache, get_default_cache
    
    text = load_file(file_path)
    
    # Configuration for caching
    profile = CHUNKING_CONFIG.profiles.get(workload_type, CHUNKING_CONFIG.profiles["mixed"])
    config = {
        "workload_type": workload_type,
        "chunk_size": profile.chunk_size,
        "overlap": profile.overlap,
        "embedding_model": EMBEDDING_CONFIG.model_name,
        "timestamp": time.time(),
    }
    
    # Generate deterministic cache key using SHA256
    if cache_name:
        index_name = cache_name
    else:
        # Use file hash + config hash for stable cache key
        from .indexing.file_fingerprint import compute_file_hash
        file_hash = compute_file_hash(file_path)
        config_str = f"{config['chunk_size']}_{config['overlap']}_{config['embedding_model']}"
        config_hash = hashlib.sha256(config_str.encode('utf-8')).hexdigest()[:16]
        index_name = f"{file_hash[:16]}_{config_hash}"
    
    # Try to load from cache
    if use_cache:
        cache = get_default_cache()
        
        cached_data = cache.load_index_cache(
            index_name=index_name,
            file_paths=[file_path],
            config=config,
        )
        
        if cached_data is not None:
            print(f"Loaded cached index from {cache.cache_dir / index_name}")
            chunks = cached_data["chunks"]
            embeddings = cached_data["embeddings"]
            store = VectorStore(embeddings)
            return store, chunks
    
    # Build index from scratch
    print("Building RAG index from scratch...")
    
    # Adaptive chunking based on centrally configured workload profile
    chunks = chunk_text(
        text,
        chunk_size=config["chunk_size"],
        overlap=config["overlap"],
        workload_type=workload_type,
    )
    
    embeddings = get_embeddings(chunks)

    store = VectorStore(embeddings)
    
    # Save to cache
    if use_cache:
        cache = get_default_cache()
        # index_name is already computed above using deterministic hashing
        
        cache.save_index_cache(
            index_name=index_name,
            file_paths=[file_path],
            chunks=chunks,
            embeddings=embeddings,
            faiss_index=store.index,
            config=config,
        )
        print(f"Saved index to cache at {cache.cache_dir / index_name}")
    
    return store, chunks


def ask_rag(
    query,
    store,
    chunks,
    reranker=None,
    context_selector=None,
    retrieval_top_k=10,
    rerank_top_n=5,
    context_top_n=3,
):
    """
    Adaptive RAG pipeline with query-driven orchestration.
    
    This pipeline uses the query understanding system to dynamically
    adapt retrieval depth, reranking, compression, and prompting based on
    query semantics.
    
    Args:
        query: User question
        store: Vector store for semantic retrieval
        chunks: Text chunks
        reranker: Optional cross-encoder reranker
        context_selector: Optional context compressor
        retrieval_top_k: Default retrieval depth (overridden by query analysis)
        rerank_top_n: Default rerank depth (overridden by query analysis)
        context_top_n: Default context chunks (overridden by query analysis)
    
    Returns:
        Tuple of (prompt, results, compressed, timings, analysis)
    """
    timings = {}
    
    # Query understanding with safe fallback
    try:
        analysis = understand_query(query)
    except Exception as e:
        # Safe fallback: if query understanding fails, use analytical as default
        from .core.types import QueryAnalysis
        from .query import build_retrieval_policy
        
        analysis = QueryAnalysis(
            query=query,
            query_type="analytical",
            confidence=0.5,
            retrieval_strategy=build_retrieval_policy("analytical"),
            answer_style="interpretation",
        )
        timings["query_analysis_error"] = str(e)
    
    config = analysis.retrieval_strategy.copy()
    config_used = {
        "query_type": analysis.query_type,
        "confidence": analysis.confidence,
        "answer_style": analysis.answer_style,
        **config
    }
    
    # Apply adaptive parameters from query analysis
    retrieval_top_k = config["retrieval_top_k"]
    rerank_top_n = config["rerank_top_n"]
    context_top_n = config["context_top_n"]
    use_reranker = config["use_reranker"]
    use_context_selector = config["use_context_selector"]
    max_new_tokens = config["max_new_tokens"]
    use_extractive_first = config["use_extractive_first"]
    query_type = analysis.query_type
    answer_style = analysis.answer_style
    
    # Store query analysis metadata in timings for analytics
    timings["query_analysis"] = {
        "query_type": analysis.query_type,
        "confidence": analysis.confidence,
        "entities": analysis.entities,
        "keywords": analysis.keywords,
        "answer_style": analysis.answer_style,
        "priority_mode": config.get("priority_mode", "balanced"),
        "ambiguity_flags": analysis.ambiguity_flags,
    }

    t0 = time.perf_counter()
    
    # Slot-aware retrieval for entity coverage
    # If query requires evidence for multiple answer slots, use slot-based retrieval
    slot_coverage_info = None
    if analysis.requires_entity_coverage and analysis.expected_answer_slots:
        # Build focused queries for each slot using anchor_entity
        slot_queries = build_slot_queries(query, analysis.expected_answer_slots, anchor_entity=analysis.anchor_entity)
        
        # Retrieve evidence for each slot
        slot_results = retrieve_for_slots(slot_queries, store, chunks, top_k_per_slot=max(2, retrieval_top_k // len(slot_queries)))
        
        # Merge and deduplicate results
        retrieved_results = merge_and_deduplicate_chunks(slot_results, max_total_chunks=retrieval_top_k)
        
        # Validate slot coverage
        slot_coverage_info = validate_slot_coverage(analysis.expected_answer_slots, retrieved_results)
        
        # Deterministic extraction: try to extract slot answers directly from evidence
        from .answering.evidence_answer_extractor import extract_factual_answer_from_evidence
        chunk_texts = [r.text for r in retrieved_results]
        extraction_result = extract_factual_answer_from_evidence(
            query=query,
            evidence_chunks=chunk_texts,
            expected_answer_slots=analysis.expected_answer_slots,
            anchor_entity=analysis.anchor_entity
        )
        
        # If all slots are extracted deterministically, use them directly
        if extraction_result.success:
            timings["entity_coverage"] = slot_coverage_info
            timings["retrieval_ms"] = (time.perf_counter() - t0) * 1000
            return extraction_result.answer, retrieved_results, timings
        
        # Store coverage info in timings
        timings["entity_coverage"] = slot_coverage_info
    else:
        # Standard retrieval for non-entity-coverage queries
        # Entity-balanced retrieval for comparative queries
        comparison_targets = analysis.comparison_targets if analysis.comparison_targets else None

        retrieved_results = retrieve(query, store, chunks, top_k=retrieval_top_k, comparison_targets=comparison_targets)
    
    timings["retrieval_ms"] = (time.perf_counter() - t0) * 1000

    # Adaptive reranking based on query type
    if use_reranker and reranker is not None:
        t0 = time.perf_counter()
        results = reranker.rerank(query=query, results=retrieved_results, top_n=rerank_top_n)
        timings["rerank_ms"] = (time.perf_counter() - t0) * 1000
    else:
        results = retrieved_results
        timings["rerank_ms"] = 0.0

    # Adaptive context selection based on query type
    if use_context_selector and context_selector is not None:
        t0 = time.perf_counter()
        compressed = context_selector.select(query, results, comparison_targets=comparison_targets)
        timings["compression_ms"] = (time.perf_counter() - t0) * 1000
        mistral_prompt = build_prompt(query, compressed.context, answer_style=answer_style, query_analysis=analysis, slot_coverage_info=slot_coverage_info)
        context_for_validation = compressed.context
    else:
        compressed = None
        timings["compression_ms"] = 0.0
        raw_context = "\n\n".join(r.text for r in results[:context_top_n])
        mistral_prompt = build_prompt(query, raw_context, answer_style=answer_style, query_analysis=analysis, slot_coverage_info=slot_coverage_info)
        context_for_validation = raw_context

    # Store adaptive config in timings for analytics
    timings["adaptive_config"] = config_used
    
    # Store prompt orchestration metadata for traceability
    timings["prompt_strategy"] = extract_prompt_metadata(analysis, answer_style)
    
    # Store context for validation
    timings["_context_for_validation"] = context_for_validation
    
    # Phase 2: Deterministic factual answer extraction
    # Try to extract answers directly from evidence before generator invocation
    deterministic_answer = None
    if should_use_deterministic_extraction(analysis):
        chunk_texts = [r.text for r in results]
        extraction_result = extract_factual_answer_from_evidence(
            query=query,
            evidence_chunks=chunk_texts,
            answer_targets=analysis.answer_targets,
            expected_answer_slots=analysis.expected_answer_slots,
            anchor_entity=analysis.anchor_entity,
        )
        
        timings["deterministic_extraction"] = {
            "attempted": True,
            "success": extraction_result.success,
            "confidence": extraction_result.confidence,
            "extracted_slots": extraction_result.extracted_slots,
            "missing_slots": extraction_result.missing_slots,
        }
        
        if extraction_result.success and extraction_result.confidence > 0.7:
            # Use deterministic answer, skip generator
            deterministic_answer = extraction_result.answer
            timings["used_deterministic_answer"] = True
            timings["deterministic_answer"] = deterministic_answer
    else:
        timings["deterministic_extraction"] = {
            "attempted": False,
            "reason": "query_not_eligible",
        }
    
    # If deterministic extraction succeeded, return early with answer
    if deterministic_answer:
        return deterministic_answer, results, timings

    return mistral_prompt, results, compressed, timings, analysis


def validate_generated_answer(
    query: str,
    answer: str,
    timings: Dict[str, Any],
    analysis: Any,
) -> tuple[str, Dict[str, Any]]:
    """
    Validate a generated answer using the answer validation layer.
    
    This function performs grounding verification, hallucination detection,
    and validation action decision. It returns the final answer (possibly
    revised or refused) and updated timings with validation metadata.
    
    Args:
        query: The original user query
        answer: The generated answer to validate
        timings: The timings dictionary from ask_rag()
        analysis: The QueryAnalysis object from ask_rag()
    
    Returns:
        Tuple of (final_answer, updated_timings)
    """
    # Extract context from timings
    context = timings.get("_context_for_validation", "")
    
    # Remove the temporary context from timings
    timings.pop("_context_for_validation", None)
    
    # Perform validation
    t0 = time.perf_counter()
    validation_result = validate_answer(
        query=query,
        answer=answer,
        context=context,
        query_analysis=analysis,
    )
    timings["validation_ms"] = (time.perf_counter() - t0) * 1000
    
    # Apply validation action
    final_answer = answer
    if validation_result.validation_action == "accept":
        final_answer = answer
    elif validation_result.validation_action == "revise":
        final_answer = revise_answer(answer, context, validation_result)
    elif validation_result.validation_action == "refuse":
        final_answer = "I don't know from the provided context."
    elif validation_result.validation_action == "clarify":
        final_answer = "Your question appears ambiguous. Could you clarify what you mean?"
    
    # Store validation metadata for traceability
    timings["validation"] = {
        "grounded": validation_result.grounded,
        "confidence": validation_result.confidence,
        "validation_action": validation_result.validation_action,
        "hallucination_detected": validation_result.hallucination_detected,
        "evidence_coverage": validation_result.evidence_coverage,
        "unsupported_claims_count": len(validation_result.unsupported_claims),
        "missing_entities_count": len(validation_result.missing_entities),
        "contradictions_count": len(validation_result.contradictions),
        "ambiguity_detected": validation_result.ambiguity_detected,
    }
    
    return final_answer, timings


def ask_rag_with_retry(
    query,
    store,
    chunks,
    reranker=None,
    context_selector=None,
    retrieval_top_k=10,
    rerank_top_n=5,
    context_top_n=3,
    enable_retry=True,
    max_retry_attempts=2,
):
    """
    Adaptive RAG pipeline with retrieval retry and self-correction loop.
    
    This pipeline extends ask_rag with automatic retry logic when answer
    validation fails or evidence is insufficient. It implements the
    self-correcting retrieval architecture.
    
    Args:
        query: User question
        store: Vector store for semantic retrieval
        chunks: Text chunks
        reranker: Optional cross-encoder reranker
        context_selector: Optional context compressor
        retrieval_top_k: Default retrieval depth (overridden by query analysis)
        rerank_top_n: Default rerank depth (overridden by query analysis)
        context_top_n: Default context chunks (overridden by query analysis)
        enable_retry: Whether to enable retry logic (default True)
        max_retry_attempts: Maximum number of retry attempts (default 2)
    
    Returns:
        Tuple of (prompt, results, compressed, timings, analysis, final_answer, retry_metadata)
    """
    # Store attempt history for best answer selection
    attempt_history: List[RetryAttempt] = []
    retry_metadata = {
        "enabled": enable_retry,
        "total_attempts": 0,
        "retry_performed": False,
        "attempts": [],
    }
    
    # Initial attempt
    attempt_number = 1
    current_query = query
    current_policy = {
        "retrieval_top_k": retrieval_top_k,
        "rerank_top_n": rerank_top_n,
        "context_top_n": context_top_n,
    }
    
    while True:
        # Execute single RAG pass
        prompt, results, compressed, timings, analysis = ask_rag(
            query=current_query,
            store=store,
            chunks=chunks,
            reranker=reranker,
            context_selector=context_selector,
            retrieval_top_k=current_policy["retrieval_top_k"],
            rerank_top_n=current_policy["rerank_top_n"],
            context_top_n=current_policy["context_top_n"],
        )
        
        # Store attempt info
        attempt_info = {
            "attempt_number": attempt_number,
            "query": current_query,
            "policy": current_policy.copy(),
            "timings": timings.copy(),
        }
        retry_metadata["attempts"].append(attempt_info)
        
        # If retry is disabled, break after first attempt
        if not enable_retry:
            break
        
        # If we've exceeded max attempts, break
        if attempt_number >= max_retry_attempts + 1:
            break
        
        # For this implementation, we need the answer to decide on retry
        # Since generation happens outside this function, we'll return
        # the first pass and let the caller handle validation and retry
        # In a full implementation, you would:
        # 1. Generate answer
        # 2. Validate answer
        # 3. Decide retry
        # 4. If retry, expand policy and loop
        # 5. Select best answer
        
        # For now, break after first attempt
        # A complete implementation would require integration with the generator
        break
    
    retry_metadata["total_attempts"] = attempt_number
    retry_metadata["retry_performed"] = attempt_number > 1
    
    # Return with placeholder for answer (to be filled by caller)
    return prompt, results, compressed, timings, analysis, None, retry_metadata


def execute_retry_loop(
    query: str,
    store,
    chunks,
    generate_fn,  # Function that takes (prompt) -> answer
    reranker=None,
    context_selector=None,
    retrieval_top_k=10,
    rerank_top_n=5,
    context_top_n=3,
    max_retry_attempts=2,
) -> Dict[str, Any]:
    """
    Execute the full retry loop with generation and validation.
    
    This function implements the complete self-correcting retrieval loop.
    
    Args:
        query: User question
        store: Vector store for semantic retrieval
        chunks: Text chunks
        generate_fn: Function that generates answer from prompt
        reranker: Optional cross-encoder reranker
        context_selector: Optional context compressor
        retrieval_top_k: Default retrieval depth
        rerank_top_n: Default rerank depth
        context_top_n: Default context chunks
        max_retry_attempts: Maximum retry attempts (default 2)
    
    Returns:
        Dictionary with complete results
    """
    attempt_history: List[RetryAttempt] = []
    retry_metadata = {
        "enabled": True,
        "total_attempts": 0,
        "retry_performed": False,
        "attempts": [],
    }
    
    current_query = query
    current_policy = {
        "retrieval_top_k": retrieval_top_k,
        "rerank_top_n": rerank_top_n,
        "context_top_n": context_top_n,
    }
    
    for attempt_number in range(1, max_retry_attempts + 2):
        # Execute RAG pass
        prompt, results, compressed, timings, analysis = ask_rag(
            query=current_query,
            store=store,
            chunks=chunks,
            reranker=reranker,
            context_selector=context_selector,
            retrieval_top_k=current_policy["retrieval_top_k"],
            rerank_top_n=current_policy["rerank_top_n"],
            context_top_n=current_policy["context_top_n"],
        )
        
        # Generate answer
        answer = generate_fn(prompt)
        
        # Validate answer
        context = timings.get("_context_for_validation", "")
        validation = validate_answer(
            query=query,
            answer=answer,
            context=context,
            query_analysis=analysis,
        )
        
        # Store attempt
        attempt = RetryAttempt(
            attempt_number=attempt_number,
            query=current_query,
            answer=answer,
            validation=validation,
            retrieval_policy=current_policy.copy(),
            timings=timings.copy(),
        )
        attempt_history.append(attempt)
        
        # Store attempt metadata
        attempt_info = {
            "attempt_number": attempt_number,
            "query": current_query,
            "answer": answer,
            "policy": current_policy.copy(),
            "validation": {
                "grounded": validation.grounded,
                "confidence": validation.confidence,
                "validation_action": validation.validation_action,
                "evidence_coverage": validation.evidence_coverage,
                "hallucination_detected": validation.hallucination_detected,
            },
        }
        retry_metadata["attempts"].append(attempt_info)
        
        # Decide whether to retry
        if attempt_number == 1:
            # First attempt: decide if retry is needed
            retry_decision = decide_retry(
                query=query,
                answer=answer,
                validation=validation,
                query_analysis=analysis,
                retrieval_results=results,
            )
            
            if not retry_decision.should_retry:
                # No retry needed, return this answer
                retry_metadata["total_attempts"] = 1
                retry_metadata["retry_performed"] = False
                timings["retry"] = {
                    "retried": False,
                    "retry_reason": retry_decision.retry_reason,
                }
                
                return {
                    "final_answer": answer,
                    "prompt": prompt,
                    "results": results,
                    "compressed": compressed,
                    "timings": timings,
                    "analysis": analysis,
                    "retry_metadata": retry_metadata,
                }
            
            # Prepare for retry
            previous_policy = current_policy.copy()
            current_policy = expand_retrieval_policy(
                current_policy,
                retry_decision.retry_strategy,
                retry_decision,
            )
            current_query = retry_decision.retry_query
            
            # Add retry metadata to timings
            timings["retry"] = build_retry_metadata(
                retry_decision,
                attempt_number + 1,
                previous_policy,
                current_policy,
            )
        else:
            # Subsequent attempts: check if should continue
            retry_decision = decide_retry(
                query=query,
                answer=answer,
                validation=validation,
                query_analysis=analysis,
                retrieval_results=results,
            )
            
            if not should_continue_retry(attempt_number, retry_decision, validation):
                # Stop retrying
                break
            
            # Prepare for next retry
            previous_policy = current_policy.copy()
            current_policy = expand_retrieval_policy(
                current_policy,
                retry_decision.retry_strategy,
                retry_decision,
            )
            current_query = retry_decision.retry_query
    
    # Select best answer from all attempts
    best_answer, best_attempt, selection_metadata = select_best_answer(attempt_history)
    
    # Update retry metadata
    retry_metadata["total_attempts"] = len(attempt_history)
    retry_metadata["retry_performed"] = len(attempt_history) > 1
    retry_metadata["best_attempt_number"] = best_attempt.attempt_number
    retry_metadata["selection_metadata"] = selection_metadata
    
    # Use the best attempt's timings and results
    final_timings = best_attempt.timings.copy()
    final_timings["retry"] = {
        "retried": True,
        "retry_count": len(attempt_history) - 1,
        "total_attempts": len(attempt_history),
        "best_attempt": best_attempt.attempt_number,
        "selection_metadata": selection_metadata,
    }
    
    return {
        "final_answer": best_answer,
        "prompt": prompt,  # Use last prompt
        "results": results,  # Use last results
        "compressed": compressed,  # Use last compressed
        "timings": final_timings,
        "analysis": analysis,
        "retry_metadata": retry_metadata,
    }


__all__ = [
    'build_slot_queries',
    'retrieve_for_slots',
    'merge_and_deduplicate_chunks',
    'validate_slot_coverage',
    'load_file',
    'build_rag',
    'ask_rag',
    'validate_generated_answer',
    'ask_rag_with_retry',
    'execute_retry_loop',
]
