"""Indexing module for chunking, embedding, vector storage, and cache persistence."""

from .chunker import chunk_text
from .embedder import get_embeddings, embed_query
from .vector_store import VectorStore
from .file_fingerprint import (
    compute_file_hash,
    compute_multiple_file_hash,
    file_has_changed,
    get_file_modification_time,
)
from .cache import IndexCache, get_default_cache

__all__ = [
    'chunk_text',
    'get_embeddings',
    'embed_query',
    'VectorStore',
    'compute_file_hash',
    'compute_multiple_file_hash',
    'file_has_changed',
    'get_file_modification_time',
    'IndexCache',
    'get_default_cache',
]
