"""Evidence-based context selector for the RAG pipeline.

Transforms reranked ``RetrievalResult`` objects into a compact,
high-density evidence context by extracting the most relevant sentences
while preserving original wording.  This is *not* summarisation; it is
retrieval-aware evidence selection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Set

from .retrieval_types import RetrievalResult, CompressedContext


_STOPWORDS = {
    "the", "a", "an", "is", "was", "were", "to", "of", "in", "on",
    "for", "with", "and", "or", "did", "who", "what", "why", "how",
    "when", "where", "does", "do", "from", "after", "before", "about",
    "as", "at", "by", "it", "its", "this", "that", "these", "those",
    "be", "been", "being", "have", "has", "had", "will", "would",
    "could", "should", "may", "might", "must", "shall", "can", "need",
}


def _words(text: str) -> Set[str]:
    """Return a set of normalised content words."""
    return {
        w for w in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if w not in _STOPWORDS and len(w) > 2
    }


def _jaccard(a: Set[str], b: Set[str]) -> float:
    """Jaccard similarity between two word sets."""
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences while preserving original wording."""
    raw = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in raw if s.strip()]


@dataclass
class _ScoredSentence:
    """Internal helper for greedy sentence selection."""

    text: str
    score: float
    chunk_id: int
    rerank_score: float | None = None


class ContextSelector:
    """Select high-density evidence sentences from retrieval results.

    Args:
        max_sentences: Hard ceiling on the number of sentences selected.
        max_chars: Hard ceiling on total character count of selected evidence.
        redundancy_threshold: Jaccard similarity above which a sentence is
            considered redundant with an already-selected sentence.
        rerank_weight: Multiplier applied to the chunk ``rerank_score`` when
            boosting sentence scores.
        hybrid_weight: Multiplier applied to the chunk ``hybrid_score`` when
            boosting sentence scores.
    """

    def __init__(
        self,
        max_sentences: int = 8,
        max_chars: int = 2500,
        redundancy_threshold: float = 0.75,
        rerank_weight: float = 0.15,
        hybrid_weight: float = 0.05,
    ) -> None:
        self.max_sentences = max_sentences
        self.max_chars = max_chars
        self.redundancy_threshold = redundancy_threshold
        self.rerank_weight = rerank_weight
        self.hybrid_weight = hybrid_weight

    def select(self, query: str, results: List[RetrievalResult]) -> CompressedContext:
        """Extract a compressed evidence context for ``query``.

        Returns a ``CompressedContext`` whose ``.context`` field is a plain
        string ready to be inserted directly into a generator prompt.
        """
        if not results:
            return CompressedContext(context="", selected_sentences=[])

        query_words = _words(query)

        # 1. Score every sentence in every result
        scored: List[_ScoredSentence] = []
        for result in results:
            sentences = _split_sentences(result.text)
            chunk_rerank = result.rerank_score if result.rerank_score is not None else 0.0
            chunk_hybrid = result.hybrid_score

            for sentence in sentences:
                sentence_words = _words(sentence)
                overlap = 0.0
                if query_words:
                    overlap = len(query_words & sentence_words) / len(query_words)

                # Boost sentences that come from highly-ranked chunks
                score = (
                    overlap
                    + chunk_rerank * self.rerank_weight
                    + chunk_hybrid * self.hybrid_weight
                )

                # Penalise very short sentences (likely fragments or headings)
                word_count = len(re.findall(r"[a-zA-Z0-9]+", sentence.lower()))
                if word_count <= 3:
                    score -= 0.1

                scored.append(
                    _ScoredSentence(
                        text=sentence,
                        score=score,
                        chunk_id=result.chunk_id,
                        rerank_score=chunk_rerank if chunk_rerank else None,
                    )
                )

        # 2. Greedy selection with redundancy filtering
        scored.sort(key=lambda s: s.score, reverse=True)

        selected: List[_ScoredSentence] = []
        selected_word_sets: List[Set[str]] = []
        total_chars = 0
        supporting_chunk_ids: set[int] = set()

        for candidate in scored:
            if len(selected) >= self.max_sentences:
                break

            if total_chars + len(candidate.text) > self.max_chars:
                break

            cand_words = _words(candidate.text)

            # Redundancy check against already-selected sentences
            redundant = False
            for existing_words in selected_word_sets:
                if _jaccard(cand_words, existing_words) > self.redundancy_threshold:
                    redundant = True
                    break

            if redundant:
                continue

            selected.append(candidate)
            selected_word_sets.append(cand_words)
            total_chars += len(candidate.text)
            supporting_chunk_ids.add(candidate.chunk_id)

        # 3. Re-sort selected sentences by chunk order for coherence
        chunk_order = {r.chunk_id: i for i, r in enumerate(results)}
        selected.sort(key=lambda s: chunk_order.get(s.chunk_id, 0))

        context = " ".join(s.text for s in selected)
        metadata = {
            "evaluated_sentences": len(scored),
            "selected_sentences": len(selected),
            "total_chars": total_chars,
            "budget_max_chars": self.max_chars,
            "budget_max_sentences": self.max_sentences,
        }

        return CompressedContext(
            context=context,
            selected_sentences=[s.text for s in selected],
            supporting_chunks=list(supporting_chunk_ids),
            metadata=metadata,
        )
