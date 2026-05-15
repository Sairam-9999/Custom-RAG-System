"""Core data structures for the RAG system.

This module centralizes all shared dataclasses used across the RAG pipeline.
Keeping types in a single location ensures consistency and avoids circular imports.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any


@dataclass
class AnswerTarget:
    """
    Structured representation of an answer target in a query.
    
    This dataclass captures semantic information about what information
    is being requested, separating the anchor entity from the target type
    and requested attribute.
    
    Attributes:
        anchor_entity: The main entity/subject (e.g., "Cooper" in "Cooper's son")
        target_type: The type of entity/relationship being queried (e.g., "son", "daughter")
        requested_attribute: The specific attribute being requested (e.g., "name", "age")
    
    Example:
        Query: "What are the names of Cooper's son and daughter?"
        
        AnswerTarget(anchor_entity="cooper", target_type="son", requested_attribute="name")
        AnswerTarget(anchor_entity="cooper", target_type="daughter", requested_attribute="name")
    """
    anchor_entity: Optional[str]
    target_type: str
    requested_attribute: Optional[str] = None


@dataclass
class QueryAnalysis:
    """
    Structured analysis of a user query.
    
    This dataclass captures all intelligence extracted from a query,
    including classification, parsing results, evidence planning, and retrieval policies.
    
    Attributes:
        query: Original user query string
        query_type: Primary classification of the query
        confidence: Confidence score for the classification (0.0 to 1.0)
        entities: Extracted named entities from the query
        keywords: Important keywords extracted from the query
        comparison_targets: Entities being compared (for comparative queries)
        time_words: Time-related expressions in the query
        question_focus: Main focus or constraint of the question
        ambiguity_flags: List of detected ambiguity signals
        requires_multiple_chunks: Whether the query needs multiple chunks for evidence
        needs_reranker: Whether the query benefits from reranking
        needs_context_selector: Whether the query needs context compression
        requires_reasoning: Whether the query requires reasoning
        requires_multi_hop: Whether the query requires multi-hop retrieval
        requires_temporal_ordering: Whether the query requires temporal ordering
        requires_entity_coverage: Whether the query requires evidence for multiple answer slots
        anchor_entity: The main entity/subject for multi-slot queries (e.g., "Cooper" in "Cooper's son and daughter")
        expected_answer_slots: List of answer slots that need evidence coverage
        answer_targets: List of structured AnswerTarget objects (semantic extraction)
        retrieval_strategy: Dict of retrieval parameters determined by query analysis
        answer_style: Preferred answer format/style
        evidence_requirements: Dict specifying what evidence the retriever must collect
        decomposition_plan: List of steps for decomposing complex queries
        metadata: Additional metadata about the query
    """
    query: str
    query_type: str
    confidence: float
    
    entities: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    
    comparison_targets: List[str] = field(default_factory=list)
    time_words: List[str] = field(default_factory=list)
    
    question_focus: str | None = None
    ambiguity_flags: List[str] = field(default_factory=list)
    
    requires_multiple_chunks: bool = False
    needs_reranker: bool = False
    needs_context_selector: bool = False
    
    requires_reasoning: bool = False
    requires_multi_hop: bool = False
    requires_temporal_ordering: bool = False
    
    requires_entity_coverage: bool = False
    anchor_entity: str | None = None
    expected_answer_slots: List[str] = field(default_factory=list)
    answer_targets: List[AnswerTarget] = field(default_factory=list)
    
    retrieval_strategy: Dict[str, Any] = field(default_factory=dict)
    
    answer_style: str = "default"
    
    evidence_requirements: Dict[str, Any] = field(default_factory=dict)
    decomposition_plan: List[str] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """A single retrieval candidate with full score provenance.

    Attributes:
        chunk_id: Index of the chunk in the original corpus list.
        text: Raw chunk text content.
        semantic_score: Normalized dense embedding score (e.g., FAISS cosine).
        bm25_score: Normalized BM25 lexical relevance score.
        hybrid_score: Fused final score used for initial ranking.
        rerank_score: Optional score assigned by a future reranker stage.
        source: Optional document identifier or filepath for citation support.
        metadata: Generic key-value bag for domain-specific or future extensions
            (e.g., chunk offset, token count, parent document ID).
    """

    chunk_id: int
    text: str

    semantic_score: float
    bm25_score: float
    hybrid_score: float

    rerank_score: Optional[float] = None

    source: Optional[str] = None

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompressedContext:
    """Dense evidence context produced by the context selector.

    Attributes:
        context: Flattened plain-text context ready for the generator prompt.
        selected_sentences: Ordered list of evidence sentences included.
        supporting_chunks: chunk_ids that contributed at least one sentence.
        metadata: Selector-specific metadata (e.g., total sentences evaluated,
            budget used, compression ratio).
    """

    context: str
    selected_sentences: List[str] = field(default_factory=list)
    supporting_chunks: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


__all__ = [
    'AnswerTarget',
    'QueryAnalysis',
    'RetrievalResult',
    'CompressedContext',
]
