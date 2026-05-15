"""Central generic configuration for the RAG system.

All defaults can be overridden with environment variables or function arguments.
Keep domain/entity-specific values out of source code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Set


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    query_instruction: str = os.getenv(
        "RAG_QUERY_INSTRUCTION",
        "Represent this question for retrieving relevant passages: ",
    )
    normalize_embeddings: bool = os.getenv("RAG_NORMALIZE_EMBEDDINGS", "true").lower() == "true"


@dataclass(frozen=True)
class ChunkProfile:
    chunk_size: int
    overlap: int


@dataclass(frozen=True)
class ChunkingConfig:
    profiles: Dict[str, ChunkProfile] = field(default_factory=lambda: {
        "fact": ChunkProfile(chunk_size=_env_int("RAG_FACT_CHUNK_SIZE", 220), overlap=_env_int("RAG_FACT_OVERLAP", 1)),
        "analytical": ChunkProfile(chunk_size=_env_int("RAG_ANALYTICAL_CHUNK_SIZE", 450), overlap=_env_int("RAG_ANALYTICAL_OVERLAP", 2)),
        "mixed": ChunkProfile(chunk_size=_env_int("RAG_MIXED_CHUNK_SIZE", 650), overlap=_env_int("RAG_MIXED_OVERLAP", 3)),
    })


@dataclass(frozen=True)
class RetrievalConfig:
    semantic_weight: float = _env_float("RAG_SEMANTIC_WEIGHT", 0.45)
    lexical_weight: float = _env_float("RAG_LEXICAL_WEIGHT", 0.55)
    semantic_candidate_multiplier: int = _env_int("RAG_SEMANTIC_CANDIDATE_MULTIPLIER", 3)
    lexical_candidate_multiplier: int = _env_int("RAG_LEXICAL_CANDIDATE_MULTIPLIER", 2)
    min_entity_results: int = _env_int("RAG_MIN_ENTITY_RESULTS", 2)
    debug: bool = os.getenv("RAG_RETRIEVAL_DEBUG", "false").lower() == "true"


@dataclass(frozen=True)
class RerankerConfig:
    model_name: str = os.getenv("RAG_RERANKER_MODEL", "BAAI/bge-reranker-base")
    max_length: int = _env_int("RAG_RERANKER_MAX_LENGTH", 512)


@dataclass(frozen=True)
class GeneratorConfig:
    ollama_url: str = os.getenv("OLLAMA_GENERATE_URL", "http://localhost:11434/api/generate")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "mistral")
    temperature: float = _env_float("RAG_GENERATION_TEMPERATURE", 0.05)
    top_p: float = _env_float("RAG_GENERATION_TOP_P", 0.9)
    num_predict: int = _env_int("RAG_GENERATION_NUM_PREDICT", 12)
    timeout: int = _env_int("RAG_GENERATION_TIMEOUT", 45)


@dataclass(frozen=True)
class ExtractionConfig:
    min_confidence: float = _env_float("RAG_EXTRACTION_MIN_CONFIDENCE", 0.60)
    bad_extractions: Set[str] = field(default_factory=lambda: {
        "your", "his", "her", "their", "my", "our",
        "the", "this", "that", "a", "an",
        "some", "any", "all", "each", "every", "both", "few", "many",
        "other", "another", "such", "what", "which", "who", "whom", "whose",
    })


@dataclass(frozen=True)
class QueryNormalizationConfig:
    # Domain-specific aliases should be supplied by the application. Default is intentionally empty.
    entity_aliases: Dict[str, str] = field(default_factory=dict)
    relation_expansions: Dict[str, List[str]] = field(default_factory=lambda: {
        "children": ["son", "daughter"],
        "kids": ["son", "daughter"],
        "parents": ["mother", "father"],
        "siblings": ["brother", "sister"],
    })


@dataclass(frozen=True)
class RuntimeConfig:
    debug: bool = os.getenv("RAG_DEBUG", "false").lower() == "true"


EMBEDDING_CONFIG = EmbeddingConfig()
CHUNKING_CONFIG = ChunkingConfig()
RETRIEVAL_CONFIG = RetrievalConfig()
GENERATOR_CONFIG = GeneratorConfig()
RERANKER_CONFIG = RerankerConfig()
EXTRACTION_CONFIG = ExtractionConfig()
QUERY_NORMALIZATION_CONFIG = QueryNormalizationConfig()
RUNTIME_CONFIG = RuntimeConfig()
CACHE_VERSION = os.getenv("RAG_CACHE_VERSION", "1.0")
