"""Cross-encoder reranker for the RAG pipeline.

Consumes structured RetrievalResult objects and produces a reordered list
with an added `rerank_score`.  The reranker is intentionally decoupled
from retrieval and generation so it can be swapped, A/B tested, or
disabled without touching downstream code.
"""

from __future__ import annotations

import copy
from typing import List

from .retrieval_types import RetrievalResult


class CrossEncoderReranker:
    """Local cross-encoder reranker using sentence-transformers.

    Args:
        model_name: HuggingFace model identifier.  Common choices:
            ``"BAAI/bge-reranker-base"`` (default, higher quality)
            ``"cross-encoder/ms-marco-MiniLM-L6-v2"`` (lighter)
        device: ``"cuda"``, ``"cpu"``, or ``None`` (auto).
        max_length: Maximum token length passed to the cross-encoder.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: str | None = None,
        max_length: int = 512,
    ) -> None:
        # Lazy import so the rest of the pipeline can be imported even
        # when sentence-transformers is not installed.
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for the reranker. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        self.model = CrossEncoder(model_name, device=device, max_length=max_length)
        self.model_name = model_name

    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_n: int = 5,
    ) -> List[RetrievalResult]:
        """Rerank ``results`` for ``query`` and return the top ``top_n``.

        The returned list is sorted by ``rerank_score`` descending.
        All original fields (``chunk_id``, ``semantic_score``,
        ``bm25_score``, ``hybrid_score``, ``source``, ``metadata``) are
        preserved.  A shallow copy of each ``RetrievalResult`` is made
        before setting ``rerank_score`` to keep debugging clean.
        """
        if not results:
            return []

        pairs = [(query, r.text) for r in results]
        scores = self.model.predict(pairs, convert_to_numpy=True)

        # Build new results with rerank_score attached
        reranked = []
        for result, score in zip(results, scores):
            # shallow copy keeps debugging clean and avoids accidental
            # mutation of the original retrieval results
            new_result = copy.copy(result)
            new_result.rerank_score = float(score)
            reranked.append(new_result)

        reranked.sort(key=lambda r: r.rerank_score or 0.0, reverse=True)
        return reranked[:top_n]
