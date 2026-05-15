from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Optional

from sentence_transformers import SentenceTransformer

from ..core.config import EMBEDDING_CONFIG


@lru_cache(maxsize=4)
def get_embedding_model(model_name: Optional[str] = None) -> SentenceTransformer:
    """Load and cache the embedding model lazily."""
    return SentenceTransformer(model_name or EMBEDDING_CONFIG.model_name)


def get_embeddings(chunks: Iterable[str], batch_size: int = 32, model_name: Optional[str] = None):
    clean_chunks = [chunk.strip() for chunk in chunks if chunk and chunk.strip()]

    if not clean_chunks:
        raise ValueError("No non-empty chunks provided for embedding.")

    model = get_embedding_model(model_name)
    return model.encode(
        clean_chunks,
        batch_size=batch_size,
        normalize_embeddings=EMBEDDING_CONFIG.normalize_embeddings,
        show_progress_bar=False,
    )


def embed_query(query: str, model_name: Optional[str] = None, query_instruction: Optional[str] = None):
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    instruction = EMBEDDING_CONFIG.query_instruction if query_instruction is None else query_instruction
    model = get_embedding_model(model_name)
    return model.encode(
        [instruction + query.strip()],
        normalize_embeddings=EMBEDDING_CONFIG.normalize_embeddings,
        show_progress_bar=False,
    )
