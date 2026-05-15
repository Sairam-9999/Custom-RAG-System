"""Evidence-based context selector for the RAG pipeline.

Transforms reranked ``RetrievalResult`` objects into a compact,
high-density evidence context by extracting the most relevant sentences
while preserving original wording.  This is *not* summarisation; it is
retrieval-aware evidence selection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Set, Optional

from ..core.types import RetrievalResult, CompressedContext


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

    def select(
        self, query: str, results: List[RetrievalResult], comparison_targets: Optional[List[str]] = None
    ) -> CompressedContext:
        """Extract a compressed evidence context for ``query``.

        Returns a ``CompressedContext`` whose ``.context`` field is a plain
        string ready to be inserted directly into a generator prompt.
        
        For comparative queries, ensures both comparison targets are represented.
        """
        if not results:
            return CompressedContext(context="", selected_sentences=[])

        query_words = _words(query)
        
        # For comparative queries, enforce entity-balanced selection
        if comparison_targets and len(comparison_targets) >= 2:
            return self._select_comparative_balanced(
                query, results, query_words, comparison_targets
            )

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

    def _select_comparative_balanced(
        self, query: str, results: List[RetrievalResult], query_words: Set[str], comparison_targets: List[str]
    ) -> CompressedContext:
        """Select sentences with entity-balanced representation for comparative queries.
        
        Ensures both comparison targets are represented in the selected context.
        """
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

                # Boost sentences that mention comparison targets
                entity_boost = 0.0
                sentence_lower = sentence.lower()
                targets_mentioned = sum(
                    1 for target in comparison_targets if target.lower() in sentence_lower
                )
                if targets_mentioned >= 2:
                    entity_boost = 0.3  # Boost sentences mentioning both entities
                elif targets_mentioned == 1:
                    entity_boost = 0.15  # Moderate boost for single entity

                # Boost sentences that come from highly-ranked chunks
                score = (
                    overlap
                    + chunk_rerank * self.rerank_weight
                    + chunk_hybrid * self.hybrid_weight
                    + entity_boost
                )

                # Penalise very short sentences
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

        # 2. Group sentences by entity presence
        entity_groups = {target: [] for target in comparison_targets}
        balanced_sentences = []  # Sentences with both entities
        
        for scored_sentence in scored:
            sentence_lower = scored_sentence.text.lower()
            targets_mentioned = [
                target for target in comparison_targets if target.lower() in sentence_lower
            ]
            
            if len(targets_mentioned) >= 2:
                balanced_sentences.append(scored_sentence)
            else:
                for target in targets_mentioned:
                    entity_groups[target].append(scored_sentence)
        
        # 3. Select with entity balancing
        selected: List[_ScoredSentence] = []
        selected_word_sets: List[Set[str]] = []
        total_chars = 0
        supporting_chunk_ids: set[int] = set()
        
        # First, prioritize balanced sentences (both entities)
        balanced_sentences.sort(key=lambda s: s.score, reverse=True)
        for candidate in balanced_sentences:
            if len(selected) >= self.max_sentences:
                break
            if total_chars + len(candidate.text) > self.max_chars:
                break
            
            cand_words = _words(candidate.text)
            
            # Redundancy check
            redundant = False
            for existing_words in selected_word_sets:
                if _jaccard(cand_words, existing_words) > self.redundancy_threshold:
                    redundant = True
                    break
            
            if not redundant:
                selected.append(candidate)
                selected_word_sets.append(cand_words)
                total_chars += len(candidate.text)
                supporting_chunk_ids.add(candidate.chunk_id)
        
        # Then, interleave entity-specific sentences to ensure balance
        entity_sentences_lists = [
            sorted(entity_groups[target], key=lambda s: s.score, reverse=True)
            for target in comparison_targets
        ]
        
        max_entity_sentences = max(len(lst) for lst in entity_sentences_lists)
        
        for i in range(max_entity_sentences):
            if len(selected) >= self.max_sentences:
                break
            
            for entity_idx, entity_sentences in enumerate(entity_sentences_lists):
                if len(selected) >= self.max_sentences:
                    break
                if i < len(entity_sentences):
                    candidate = entity_sentences[i]
                    if total_chars + len(candidate.text) > self.max_chars:
                        continue
                    
                    cand_words = _words(candidate.text)
                    
                    # Redundancy check
                    redundant = False
                    for existing_words in selected_word_sets:
                        if _jaccard(cand_words, existing_words) > self.redundancy_threshold:
                            redundant = True
                            break
                    
                    if not redundant:
                        selected.append(candidate)
                        selected_word_sets.append(cand_words)
                        total_chars += len(candidate.text)
                        supporting_chunk_ids.add(candidate.chunk_id)
        
        # Check if both entities are represented
        selected_text_lower = " ".join(s.text for s in selected).lower()
        missing_entities = [
            target for target in comparison_targets if target.lower() not in selected_text_lower
        ]
        
        # If entity missing, try to add at least one sentence for it
        if missing_entities and len(selected) < self.max_sentences:
            for missing_entity in missing_entities:
                # Find best sentence for missing entity
                for entity_sentences in entity_sentences_lists:
                    for candidate in entity_sentences:
                        if missing_entity.lower() in candidate.text.lower():
                            if total_chars + len(candidate.text) <= self.max_chars:
                                selected.append(candidate)
                                total_chars += len(candidate.text)
                                supporting_chunk_ids.add(candidate.chunk_id)
                                break
                    break
                if len(selected) >= self.max_sentences:
                    break
        
        # 4. Re-sort selected sentences by chunk order for coherence
        chunk_order = {r.chunk_id: i for i, r in enumerate(results)}
        selected.sort(key=lambda s: chunk_order.get(s.chunk_id, 0))

        context = " ".join(s.text for s in selected)
        metadata = {
            "evaluated_sentences": len(scored),
            "selected_sentences": len(selected),
            "total_chars": total_chars,
            "budget_max_chars": self.max_chars,
            "budget_max_sentences": self.max_sentences,
            "comparison_targets": comparison_targets,
            "entity_balance_check": {
                "missing_entities": missing_entities,
                "all_targets_represented": len(missing_entities) == 0,
            },
        }

        return CompressedContext(
            context=context,
            selected_sentences=[s.text for s in selected],
            supporting_chunks=list(supporting_chunk_ids),
            metadata=metadata,
        )
