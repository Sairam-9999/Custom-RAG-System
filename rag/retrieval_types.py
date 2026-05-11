"""Structured retrieval objects for the RAG pipeline.

This module defines the core data contract between the retrieval layer
and downstream consumers (rerankers, generators, evaluators, etc.).
Keeping the schema lightweight and generic ensures the architecture
remains reusable across domains and extensible for future stages
without requiring cascading changes.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


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
