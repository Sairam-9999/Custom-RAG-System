"""Retrieval evaluation framework.

This module provides a generic, reusable evaluation framework for measuring
retrieval quality across different pipeline stages (raw retrieval, reranking,
context selection).

Key features:
- Standard IR metrics: precision@k, recall@k, MRR, hit@k
- Evidence coverage scoring for content-based evaluation
- Per-stage evaluation (raw, reranked, selected)
- Aggregate statistics and failure analysis
- Generic design with no domain-specific logic

Example usage:
    from rag.evaluation import (
        RetrievalEvalCase,
        evaluate_retrieval_suite,
    )
    
    # Define test cases
    test_cases = [
        RetrievalEvalCase(
            query="What is the Tesseract?",
            expected_chunk_ids={42, 57},
            expected_evidence_strings={"Tesseract", "three-dimensional"},
        ),
    ]
    
    # Run evaluation
    summary = evaluate_retrieval_suite(
        test_cases=test_cases,
        query=query_fn,
        store=vector_store,
        chunks=chunks,
        reranker=reranker,
        context_selector=context_selector,
        verbose=True,
    )
    
    # Check results
    print(f"Success rate: {summary.success_rate:.1%}")
    print(f"Average MRR: {summary.avg_mrr:.3f}")
"""

from .eval_types import (
    RetrievalEvalCase,
    StageMetrics,
    RetrievalEvalResult,
    RetrievalEvalSummary,
)
from .retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    hit_at_k,
    mean_reciprocal_rank,
    evidence_coverage,
    compute_all_metrics,
)
from .evaluator import (
    evaluate_retrieval_case,
    evaluate_retrieval_suite,
)

__all__ = [
    # Types
    'RetrievalEvalCase',
    'StageMetrics',
    'RetrievalEvalResult',
    'RetrievalEvalSummary',
    # Metrics
    'precision_at_k',
    'recall_at_k',
    'hit_at_k',
    'mean_reciprocal_rank',
    'evidence_coverage',
    'compute_all_metrics',
    # Evaluator
    'evaluate_retrieval_case',
    'evaluate_retrieval_suite',
]
