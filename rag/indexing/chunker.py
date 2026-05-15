from __future__ import annotations

import re
from typing import Optional

from ..core.config import CHUNKING_CONFIG


def split_sentences(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(
    text: str,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
    workload_type: str = "mixed",
    is_fact_query: Optional[bool] = None,
) -> list[str]:
    """Chunk text using configurable workload profiles instead of hardcoded sizes."""
    if not text or not text.strip():
        return []

    if is_fact_query is True:
        workload_type = "fact"

    profile = CHUNKING_CONFIG.profiles.get(workload_type, CHUNKING_CONFIG.profiles["mixed"])
    chunk_size = profile.chunk_size if chunk_size is None else chunk_size
    overlap = profile.overlap if overlap is None else overlap
    overlap = max(0, overlap)

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []

    for para in paragraphs:
        sentences = split_sentences(para)
        current: list[str] = []

        for sentence in sentences:
            temp = " ".join(current + [sentence])
            if len(temp) <= chunk_size:
                current.append(sentence)
            else:
                if current:
                    chunks.append(" ".join(current))
                current = current[-overlap:] + [sentence] if overlap else [sentence]

        if current:
            chunks.append(" ".join(current))

    return chunks
