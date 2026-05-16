"""Evaluation types for retrieval evaluation framework.

This module defines dataclasses for test cases, results, and summaries
used in evaluating retrieval quality across different pipeline stages.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set


@dataclass
class RetrievalEvalCase:
    """
    A single test case for retrieval evaluation.
    
    Each test case specifies what should be retrieved for a given query.
    Expected relevant items can be specified either by chunk IDs or by
    evidence strings that should appear in the retrieved content.
    
    Attributes:
        query: The query string to test
        expected_chunk_ids: Set of chunk IDs that should be retrieved (optional)
        expected_evidence_strings: Set of strings that should appear in retrieved chunks (optional)
        query_type: Optional query type classification for filtering/grouping
        expected_entities: Optional list of entities that should be covered
        notes: Optional notes about this test case
        metadata: Additional metadata for this test case
    """
    query: str
    expected_chunk_ids: Optional[Set[int]] = None
    expected_evidence_strings: Optional[Set[str]] = None
    query_type: Optional[str] = None
    expected_entities: Optional[List[str]] = None
    notes: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Ensure at least one expectation type is provided."""
        if self.expected_chunk_ids is None:
            self.expected_chunk_ids = set()
        if self.expected_evidence_strings is None:
            self.expected_evidence_strings = set()
        
        if not self.expected_chunk_ids and not self.expected_evidence_strings:
            raise ValueError(
                "RetrievalEvalCase must have either expected_chunk_ids or "
                "expected_evidence_strings (or both)"
            )


@dataclass
class StageMetrics:
    """
    Metrics for a single pipeline stage.
    
    Attributes:
        retrieved_chunk_ids: List of chunk IDs retrieved at this stage
        retrieved_texts: List of text contents retrieved at this stage
        precision_at_k: Precision@k for various k values
        recall_at_k: Recall@k for various k values
        hit_at_k: Hit@k for various k values
        mrr: Mean reciprocal rank
        evidence_coverage: Fraction of expected evidence strings found
        missing_chunk_ids: Expected chunk IDs not retrieved
        missing_evidence: Expected evidence strings not found
        retrieved_but_unused: Chunk IDs retrieved but not in expected set
    """
    retrieved_chunk_ids: List[int] = field(default_factory=list)
    retrieved_texts: List[str] = field(default_factory=list)
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    hit_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    evidence_coverage: float = 0.0
    missing_chunk_ids: Set[int] = field(default_factory=set)
    missing_evidence: Set[str] = field(default_factory=set)
    retrieved_but_unused: Set[int] = field(default_factory=set)


@dataclass
class RetrievalEvalResult:
    """
    Evaluation result for a single test case across all pipeline stages.
    
    This captures the performance of retrieval, reranking, and context
    selection stages for a single query.
    
    Attributes:
        test_case: The original test case
        raw_retrieval: Metrics for the initial retrieval stage
        reranked_retrieval: Metrics for the reranked stage (optional)
        selected_evidence: Metrics for the context-selected evidence (optional)
        success: Whether the query was successfully answered (based on coverage)
        failure_reason: If not successful, the primary reason for failure
        metadata: Additional metadata about this evaluation
    """
    test_case: RetrievalEvalCase
    raw_retrieval: StageMetrics
    reranked_retrieval: Optional[StageMetrics] = None
    selected_evidence: Optional[StageMetrics] = None
    success: bool = False
    failure_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalEvalSummary:
    """
    Aggregated summary statistics for a full evaluation suite.
    
    Attributes:
        total_cases: Total number of test cases evaluated
        successful_cases: Number of cases with successful retrieval
        success_rate: Fraction of cases with successful retrieval
        avg_precision_at_k: Average precision@k across all cases
        avg_recall_at_k: Average recall@k across all cases
        avg_hit_at_k: Average hit@k across all cases
        avg_mrr: Average mean reciprocal rank across all cases
        avg_evidence_coverage: Average evidence coverage across all cases
        failed_cases: List of failed test case identifiers
        failure_reasons: Count of cases by failure reason
        stage_comparison: Comparison of metrics across stages (raw vs reranked vs selected)
        metadata: Additional metadata about the evaluation run
    """
    total_cases: int = 0
    successful_cases: int = 0
    success_rate: float = 0.0
    
    avg_precision_at_k: Dict[int, float] = field(default_factory=dict)
    avg_recall_at_k: Dict[int, float] = field(default_factory=dict)
    avg_hit_at_k: Dict[int, float] = field(default_factory=dict)
    avg_mrr: float = 0.0
    avg_evidence_coverage: float = 0.0
    
    failed_cases: List[str] = field(default_factory=list)
    failure_reasons: Dict[str, int] = field(default_factory=dict)
    
    stage_comparison: Dict[str, Dict[str, float]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


__all__ = [
    'RetrievalEvalCase',
    'StageMetrics',
    'RetrievalEvalResult',
    'RetrievalEvalSummary',
]
