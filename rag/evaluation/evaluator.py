"""Retrieval evaluation framework.

This module provides the main evaluator for testing retrieval quality across
different pipeline stages (raw retrieval, reranking, context selection).
"""

from typing import List, Optional, Dict, Any
from .eval_types import (
    RetrievalEvalCase,
    RetrievalEvalResult,
    RetrievalEvalSummary,
    StageMetrics,
)
from .retrieval_metrics import compute_all_metrics


def _build_stage_metrics(
    retrieved_ids: List[int],
    retrieved_texts: List[str],
    relevant_ids: set,
    expected_evidence: set,
    k_values: List[int] = [1, 3, 5, 10],
) -> StageMetrics:
    """
    Build StageMetrics from retrieval results and expectations.
    
    Args:
        retrieved_ids: List of retrieved chunk IDs in order
        retrieved_texts: List of retrieved text contents
        relevant_ids: Set of relevant chunk IDs
        expected_evidence: Set of evidence strings that should appear
        k_values: List of k values for precision@k, recall@k, hit@k
        
    Returns:
        StageMetrics object with all computed metrics
    """
    metrics = compute_all_metrics(
        retrieved_ids=retrieved_ids,
        retrieved_texts=retrieved_texts,
        relevant_ids=relevant_ids,
        expected_evidence=expected_evidence,
        k_values=k_values,
    )
    
    return StageMetrics(
        retrieved_chunk_ids=retrieved_ids,
        retrieved_texts=retrieved_texts,
        precision_at_k=metrics["precision_at_k"],
        recall_at_k=metrics["recall_at_k"],
        hit_at_k=metrics["hit_at_k"],
        mrr=metrics["mrr"],
        evidence_coverage=metrics["evidence_coverage"],
        missing_chunk_ids=metrics["missing_chunk_ids"],
        missing_evidence=metrics["missing_evidence"],
        retrieved_but_unused=metrics["retrieved_but_unused"],
    )


def evaluate_retrieval_case(
    test_case: RetrievalEvalCase,
    query,
    store,
    chunks,
    reranker=None,
    context_selector=None,
    retrieval_policy: Optional[Dict[str, Any]] = None,
    k_values: List[int] = [1, 3, 5, 10],
    verbose: bool = False,
) -> RetrievalEvalResult:
    """
    Evaluate a single retrieval test case across pipeline stages.
    
    This function executes the retrieval pipeline for a single query and
    evaluates the quality of results at each stage: raw retrieval,
    reranking (if provided), and context selection (if provided).
    
    Args:
        test_case: The test case to evaluate
        query: Query understanding function or query string
        store: Vector store for semantic retrieval
        chunks: List of document chunks
        reranker: Optional cross-encoder reranker
        context_selector: Optional context compressor
        retrieval_policy: Optional retrieval policy dict (top_k, etc.)
        k_values: List of k values for precision@k, recall@k, hit@k
        verbose: Whether to print detailed diagnostic information
        
    Returns:
        RetrievalEvalResult with metrics for all evaluated stages
    """
    from rag.retrieval import retrieve
    from rag.core.types import RetrievalResult
    
    # Set default retrieval policy
    if retrieval_policy is None:
        retrieval_policy = {
            "retrieval_top_k": 10,
            "rerank_top_n": 5,
            "context_top_n": 3,
        }
    
    # Normalize query to string if it's a function
    query_str = test_case.query if isinstance(test_case.query, str) else test_case.query
    
    # Extract retrieval parameters
    retrieval_top_k = retrieval_policy.get("retrieval_top_k", 10)
    rerank_top_n = retrieval_policy.get("rerank_top_n", 5)
    
    # Stage 1: Raw retrieval
    raw_results = retrieve(
        query=query_str,
        store=store,
        chunks=chunks,
        top_k=retrieval_top_k,
    )
    
    raw_ids = [r.chunk_id for r in raw_results]
    raw_texts = [r.text for r in raw_results]
    
    raw_metrics = _build_stage_metrics(
        retrieved_ids=raw_ids,
        retrieved_texts=raw_texts,
        relevant_ids=test_case.expected_chunk_ids,
        expected_evidence=test_case.expected_evidence_strings,
        k_values=k_values,
    )
    
    if verbose:
        print(f"\n=== Raw Retrieval for: {query_str} ===")
        print(f"Retrieved {len(raw_ids)} chunks")
        print(f"Precision@5: {raw_metrics.precision_at_k.get(5, 0):.3f}")
        print(f"Recall@10: {raw_metrics.recall_at_k.get(10, 0):.3f}")
        print(f"MRR: {raw_metrics.mrr:.3f}")
        print(f"Evidence coverage: {raw_metrics.evidence_coverage:.3f}")
        if raw_metrics.missing_chunk_ids:
            print(f"Missing chunk IDs: {raw_metrics.missing_chunk_ids}")
        if raw_metrics.missing_evidence:
            print(f"Missing evidence: {raw_metrics.missing_evidence}")
    
    # Stage 2: Reranking (if reranker provided)
    reranked_metrics = None
    reranked_results = raw_results
    
    if reranker is not None:
        reranked_results = reranker.rerank(
            query=query_str,
            results=raw_results,
            top_n=rerank_top_n,
        )
        
        reranked_ids = [r.chunk_id for r in reranked_results]
        reranked_texts = [r.text for r in reranked_results]
        
        reranked_metrics = _build_stage_metrics(
            retrieved_ids=reranked_ids,
            retrieved_texts=reranked_texts,
            relevant_ids=test_case.expected_chunk_ids,
            expected_evidence=test_case.expected_evidence_strings,
            k_values=k_values,
        )
        
        if verbose:
            print(f"\n=== Reranked Retrieval ===")
            print(f"Reranked to {len(reranked_ids)} chunks")
            print(f"Precision@5: {reranked_metrics.precision_at_k.get(5, 0):.3f}")
            print(f"Recall@5: {reranked_metrics.recall_at_k.get(5, 0):.3f}")
            print(f"MRR: {reranked_metrics.mrr:.3f}")
            print(f"Evidence coverage: {reranked_metrics.evidence_coverage:.3f}")
    
    # Stage 3: Context selection (if context_selector provided)
    selected_metrics = None
    if context_selector is not None and reranked_results:
        compressed = context_selector.select(
            query=query_str,
            results=reranked_results,
        )
        
        selected_ids = compressed.supporting_chunks
        selected_texts = compressed.selected_sentences
        
        selected_metrics = _build_stage_metrics(
            retrieved_ids=selected_ids,
            retrieved_texts=selected_texts,
            relevant_ids=test_case.expected_chunk_ids,
            expected_evidence=test_case.expected_evidence_strings,
            k_values=k_values,
        )
        
        if verbose:
            print(f"\n=== Selected Evidence ===")
            print(f"Selected {len(selected_ids)} chunks")
            print(f"Evidence coverage: {selected_metrics.evidence_coverage:.3f}")
            if selected_metrics.missing_evidence:
                print(f"Missing evidence: {selected_metrics.missing_evidence}")
    
    # Determine overall success
    # Success is defined as having adequate evidence coverage
    # Use the best coverage across stages
    best_coverage = raw_metrics.evidence_coverage
    if reranked_metrics and reranked_metrics.evidence_coverage > best_coverage:
        best_coverage = reranked_metrics.evidence_coverage
    if selected_metrics and selected_metrics.evidence_coverage > best_coverage:
        best_coverage = selected_metrics.evidence_coverage
    
    # Success threshold: at least 80% evidence coverage
    success_threshold = 0.8
    success = best_coverage >= success_threshold
    
    failure_reason = None
    if not success:
        if best_coverage == 0.0:
            failure_reason = "no_evidence_found"
        elif best_coverage < 0.5:
            failure_reason = "insufficient_evidence"
        else:
            failure_reason = "partial_evidence"
    
    result = RetrievalEvalResult(
        test_case=test_case,
        raw_retrieval=raw_metrics,
        reranked_retrieval=reranked_metrics,
        selected_evidence=selected_metrics,
        success=success,
        failure_reason=failure_reason,
        metadata={
            "query_str": query_str,
            "retrieval_policy": retrieval_policy,
            "best_coverage": best_coverage,
            "success_threshold": success_threshold,
        },
    )
    
    return result


def evaluate_retrieval_suite(
    test_cases: List[RetrievalEvalCase],
    query,
    store,
    chunks,
    reranker=None,
    context_selector=None,
    retrieval_policy: Optional[Dict[str, Any]] = None,
    k_values: List[int] = [1, 3, 5, 10],
    verbose: bool = False,
) -> RetrievalEvalSummary:
    """
    Evaluate a suite of retrieval test cases.
    
    This function runs evaluate_retrieval_case for each test case and
    aggregates the results into summary statistics.
    
    Args:
        test_cases: List of test cases to evaluate
        query: Query understanding function or query string
        store: Vector store for semantic retrieval
        chunks: List of document chunks
        reranker: Optional cross-encoder reranker
        context_selector: Optional context compressor
        retrieval_policy: Optional retrieval policy dict (top_k, etc.)
        k_values: List of k values for precision@k, recall@k, hit@k
        verbose: Whether to print detailed diagnostic information
        
    Returns:
        RetrievalEvalSummary with aggregated metrics across all test cases
    """
    results = []
    
    for i, test_case in enumerate(test_cases):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Evaluating test case {i+1}/{len(test_cases)}: {test_case.query}")
            print(f"{'='*60}")
        
        result = evaluate_retrieval_case(
            test_case=test_case,
            query=query,
            store=store,
            chunks=chunks,
            reranker=reranker,
            context_selector=context_selector,
            retrieval_policy=retrieval_policy,
            k_values=k_values,
            verbose=verbose,
        )
        results.append(result)
    
    # Aggregate metrics
    total_cases = len(results)
    successful_cases = sum(1 for r in results if r.success)
    success_rate = successful_cases / total_cases if total_cases > 0 else 0.0
    
    # Initialize aggregate dictionaries
    avg_precision_at_k = {k: 0.0 for k in k_values}
    avg_recall_at_k = {k: 0.0 for k in k_values}
    avg_hit_at_k = {k: 0.0 for k in k_values}
    
    total_mrr = 0.0
    total_evidence_coverage = 0.0
    
    # Track metrics for each stage
    stage_metrics = {
        "raw": {"mrr": [], "coverage": []},
        "reranked": {"mrr": [], "coverage": []},
        "selected": {"mrr": [], "coverage": []},
    }
    
    failed_cases = []
    failure_reasons = {}
    
    for result in results:
        # Aggregate raw retrieval metrics
        raw = result.raw_retrieval
        for k in k_values:
            avg_precision_at_k[k] += raw.precision_at_k.get(k, 0)
            avg_recall_at_k[k] += raw.recall_at_k.get(k, 0)
            avg_hit_at_k[k] += raw.hit_at_k.get(k, 0)
        
        total_mrr += raw.mrr
        total_evidence_coverage += raw.evidence_coverage
        
        stage_metrics["raw"]["mrr"].append(raw.mrr)
        stage_metrics["raw"]["coverage"].append(raw.evidence_coverage)
        
        # Aggregate reranked metrics if available
        if result.reranked_retrieval:
            stage_metrics["reranked"]["mrr"].append(result.reranked_retrieval.mrr)
            stage_metrics["reranked"]["coverage"].append(
                result.reranked_retrieval.evidence_coverage
            )
        
        # Aggregate selected metrics if available
        if result.selected_evidence:
            stage_metrics["selected"]["mrr"].append(result.selected_evidence.mrr)
            stage_metrics["selected"]["coverage"].append(
                result.selected_evidence.evidence_coverage
            )
        
        # Track failures
        if not result.success:
            failed_cases.append(result.test_case.query)
            if result.failure_reason:
                failure_reasons[result.failure_reason] = (
                    failure_reasons.get(result.failure_reason, 0) + 1
                )
    
    # Compute averages
    for k in k_values:
        avg_precision_at_k[k] /= total_cases if total_cases > 0 else 1
        avg_recall_at_k[k] /= total_cases if total_cases > 0 else 1
        avg_hit_at_k[k] /= total_cases if total_cases > 0 else 1
    
    avg_mrr = total_mrr / total_cases if total_cases > 0 else 0.0
    avg_evidence_coverage = total_evidence_coverage / total_cases if total_cases > 0 else 0.0
    
    # Build stage comparison
    stage_comparison = {}
    for stage_name, metrics in stage_metrics.items():
        if metrics["mrr"]:
            stage_comparison[stage_name] = {
                "avg_mrr": sum(metrics["mrr"]) / len(metrics["mrr"]),
                "avg_coverage": sum(metrics["coverage"]) / len(metrics["coverage"]),
            }
    
    summary = RetrievalEvalSummary(
        total_cases=total_cases,
        successful_cases=successful_cases,
        success_rate=success_rate,
        avg_precision_at_k=avg_precision_at_k,
        avg_recall_at_k=avg_recall_at_k,
        avg_hit_at_k=avg_hit_at_k,
        avg_mrr=avg_mrr,
        avg_evidence_coverage=avg_evidence_coverage,
        failed_cases=failed_cases,
        failure_reasons=failure_reasons,
        stage_comparison=stage_comparison,
        metadata={
            "k_values": k_values,
            "retrieval_policy": retrieval_policy,
            "has_reranker": reranker is not None,
            "has_context_selector": context_selector is not None,
        },
    )
    
    if verbose:
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total cases: {total_cases}")
        print(f"Successful: {successful_cases} ({success_rate:.1%})")
        print(f"Average MRR: {avg_mrr:.3f}")
        print(f"Average evidence coverage: {avg_evidence_coverage:.3f}")
        print(f"\nPrecision@k: {', '.join(f'@{k}: {avg_precision_at_k[k]:.3f}' for k in sorted(k_values))}")
        print(f"Recall@k: {', '.join(f'@{k}: {avg_recall_at_k[k]:.3f}' for k in sorted(k_values))}")
        
        if stage_comparison:
            print(f"\nStage Comparison:")
            for stage, metrics in stage_comparison.items():
                print(f"  {stage}: MRR={metrics['avg_mrr']:.3f}, Coverage={metrics['avg_coverage']:.3f}")
        
        if failed_cases:
            print(f"\nFailed cases ({len(failed_cases)}):")
            for case in failed_cases:
                print(f"  - {case}")
        
        if failure_reasons:
            print(f"\nFailure reasons:")
            for reason, count in failure_reasons.items():
                print(f"  - {reason}: {count}")
    
    return summary


__all__ = [
    'evaluate_retrieval_case',
    'evaluate_retrieval_suite',
]
