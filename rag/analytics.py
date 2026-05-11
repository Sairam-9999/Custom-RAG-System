"""Structured analytics and evaluation logging for the RAG pipeline.

Produces newline-delimited JSONL experiment logs for offline analysis,
RAGAS evaluation, retrieval debugging, and A/B testing.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .retrieval_types import CompressedContext, RetrievalResult


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _refused(answer: str) -> bool:
    return "i don't know from the provided context" in answer.lower()


@dataclass
class RunRecord:
    """Structured record for a single RAG query execution.

    This is the canonical schema for JSONL log entries.  Fields are
    flat enough for pandas / RAGAS but grouped by pipeline stage for
    human readability.
    """

    timestamp: str
    experiment: str
    query: str
    mode: str

    retrieval: Dict[str, Any] = field(default_factory=dict)
    reranking: Dict[str, Any] = field(default_factory=dict)
    compression: Dict[str, Any] = field(default_factory=dict)
    generation: Dict[str, Any] = field(default_factory=dict)
    latency: Dict[str, float] = field(default_factory=dict)


class AnalyticsLogger:
    """Append-only JSONL logger for RAG experiment runs.

    Args:
        log_path: Path to the JSONL file.  Parent directories are created
            automatically.
        experiment: Experiment tag written into every record.  Enables
            filtering and A/B comparison across runs.
    """

    def __init__(
        self,
        log_path: str | Path = "logs/rag_runs.jsonl",
        experiment: str = "default",
    ) -> None:
        self.log_path = Path(log_path)
        self.experiment = experiment
        os.makedirs(self.log_path.parent, exist_ok=True)

    def log(
        self,
        *,
        query: str,
        mode: str,
        retrieval_results: List[RetrievalResult],
        compressed: Optional[CompressedContext] = None,
        answer: str,
        retrieval_top_k: int,
        reranker_enabled: bool,
        rerank_top_n: int,
        selector_enabled: bool,
        latency_ms: Dict[str, float],
    ) -> RunRecord:
        """Build a structured record and append it to the JSONL log.

        Returns the record for optional immediate inspection.
        """
        # Retrieval metadata — scores only, no full text to keep logs lean
        retrieval_meta: Dict[str, Any] = {
            "top_k": retrieval_top_k,
            "num_results": len(retrieval_results),
            "results": [
                {
                    "chunk_id": r.chunk_id,
                    "semantic_score": round(r.semantic_score, 4),
                    "bm25_score": round(r.bm25_score, 4),
                    "hybrid_score": round(r.hybrid_score, 4),
                    "rerank_score": (
                        round(r.rerank_score, 4) if r.rerank_score is not None else None
                    ),
                }
                for r in retrieval_results
            ],
        }

        # Reranking metadata
        reranking_meta: Dict[str, Any] = {
            "enabled": reranker_enabled,
            "top_n": rerank_top_n,
        }
        if reranker_enabled and retrieval_results:
            scores = [
                r.rerank_score for r in retrieval_results if r.rerank_score is not None
            ]
            reranking_meta["scores"] = [round(s, 4) for s in scores]
            if scores:
                reranking_meta["score_min"] = round(min(scores), 4)
                reranking_meta["score_max"] = round(max(scores), 4)
                reranking_meta["score_mean"] = round(sum(scores) / len(scores), 4)

        # Compression metadata
        compression_meta: Dict[str, Any] = {"enabled": selector_enabled}
        if selector_enabled and compressed is not None:
            total_before = sum(len(r.text) for r in retrieval_results)
            total_after = len(compressed.context)
            ratio = round(total_after / total_before, 4) if total_before else 1.0

            compression_meta.update(
                {
                    "selected_sentences": len(compressed.selected_sentences),
                    "supporting_chunks": compressed.supporting_chunks,
                    "total_chars_before": total_before,
                    "total_chars_after": total_after,
                    "compression_ratio": ratio,
                }
            )
            # Merge selector internal metadata (evaluated_sentences, etc.)
            compression_meta.update(compressed.metadata)

        # Generation metadata
        generation_meta: Dict[str, Any] = {
            "answer_preview": answer[:200],
            "refused": _refused(answer),
            "answer_length": len(answer),
        }

        record = RunRecord(
            timestamp=_now(),
            experiment=self.experiment,
            query=query,
            mode=mode,
            retrieval=retrieval_meta,
            reranking=reranking_meta,
            compression=compression_meta,
            generation=generation_meta,
            latency=latency_ms,
        )

        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record.__dict__, default=str, ensure_ascii=False) + "\n")

        return record

    def print_summary(self, record: RunRecord) -> None:
        """Print a concise terminal summary of a single run."""
        print("\n--- Analytics Summary ---\n")
        print(f"Experiment:      {record.experiment}")
        print(f"Mode:            {record.mode}")
        print(
            f"Retrieval:       {record.retrieval.get('num_results', 0)} results "
            f"(top_k={record.retrieval.get('top_k', '?')})"
        )
        print(
            f"Reranking:       {'enabled' if record.reranking.get('enabled') else 'disabled'} "
            f"(top_n={record.reranking.get('top_n', '?')})"
        )

        comp = record.compression
        if comp.get("enabled"):
            print(
                f"Compression:     {comp.get('selected_sentences', '?')} sentences, "
                f"ratio={comp.get('compression_ratio', '?')}"
            )
        else:
            print("Compression:     disabled")

        print(f"Refused:         {record.generation.get('refused', '?')}")
        print()

        lat = record.latency
        for key in ("retrieval_ms", "rerank_ms", "compression_ms", "generation_ms", "total_ms"):
            val = lat.get(key)
            if val is not None:
                print(f"{key:18s} {val:.1f} ms")
        print("--- End Analytics ---\n")
