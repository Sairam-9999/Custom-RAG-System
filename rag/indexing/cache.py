"""
Index Persistence and Cache Module

This module provides production-grade index persistence with SHA256 fingerprinting
and cache validation. It allows the RAG system to avoid rebuilding embeddings,
chunks, and indexes for unchanged files.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import faiss

from .file_fingerprint import compute_file_hash, compute_multiple_file_hash
from ..core.config import EMBEDDING_CONFIG, CACHE_VERSION


class IndexCache:
    """
    Manages index persistence and cache validation.
    
    This class handles:
    - Saving and loading FAISS indexes
    - Saving and loading chunk data
    - Metadata tracking with file hashes
    - Cache validation based on source file changes
    - Config-aware invalidation
    """
    
    def __init__(self, cache_dir: str = "cache/indexes"):
        """
        Initialize the index cache manager.
        
        Args:
            cache_dir: Directory to store cached indexes
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_cache_path(self, index_name: str) -> Path:
        """
        Get the cache directory for a specific index.
        
        Args:
            index_name: Name/identifier for the index
            
        Returns:
            Path to the cache directory
        """
        index_cache_dir = self.cache_dir / index_name
        index_cache_dir.mkdir(parents=True, exist_ok=True)
        return index_cache_dir
    
    def save_index_cache(
        self,
        index_name: str,
        file_paths: List[str],
        chunks: List[str],
        embeddings: np.ndarray,
        faiss_index: faiss.Index,
        config: Dict[str, Any],
    ) -> None:
        """
        Save index data to cache with metadata.
        
        Args:
            index_name: Name/identifier for the index
            file_paths: List of source file paths
            chunks: List of text chunks
            embeddings: Embedding matrix
            faiss_index: FAISS index object
            config: Configuration used to build the index
        """
        cache_dir = self.get_cache_path(index_name)
        
        # Compute file hashes
        file_hashes = {fp: compute_file_hash(fp) for fp in file_paths}
        combined_hash = compute_multiple_file_hash(file_paths)
        
        # Save metadata
        metadata = {
            "source_file": file_paths[0] if len(file_paths) == 1 else file_paths,
            "file_hash": combined_hash,
            "file_hashes": file_hashes,
            "num_chunks": len(chunks),
            "embedding_dim": embeddings.shape[1],
            "chunk_size": config.get("chunk_size", 650),
            "overlap": config.get("overlap", 3),
            "embedding_model": config.get("embedding_model", EMBEDDING_CONFIG.model_name),
            "created_timestamp": config.get("timestamp"),
            "cache_version": CACHE_VERSION,
            "config": config,
        }
        
        with open(cache_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Save chunks
        with open(cache_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2)
        
        # Save FAISS index
        faiss.write_index(faiss_index, str(cache_dir / "index.faiss"))
        
        # Save embeddings
        with open(cache_dir / "embeddings.npy", "wb") as f:
            np.save(f, embeddings)
    
    def load_index_cache(
        self,
        index_name: str,
        file_paths: List[str],
        config: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Load index data from cache if valid.
        
        Args:
            index_name: Name/identifier for the index
            file_paths: List of source file paths
            config: Current configuration
            
        Returns:
            Dictionary with chunks, embeddings, and faiss_index if cache is valid,
            None otherwise
        """
        cache_dir = self.get_cache_path(index_name)
        
        # Check if cache files exist
        if not (cache_dir / "metadata.json").exists():
            return None
        
        # Load metadata
        with open(cache_dir / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        # Validate cache
        if not self.cache_is_valid(file_paths, config, metadata):
            return None
        
        try:
            # Load chunks
            with open(cache_dir / "chunks.json", "r", encoding="utf-8") as f:
                chunks = json.load(f)
            
            # Load embeddings
            with open(cache_dir / "embeddings.npy", "rb") as f:
                embeddings = np.load(f)
            
            # Load FAISS index
            faiss_index = faiss.read_index(str(cache_dir / "index.faiss"))
            
            return {
                "chunks": chunks,
                "embeddings": embeddings,
                "faiss_index": faiss_index,
                "metadata": metadata,
            }
        except Exception as e:
            # If loading fails, treat as invalid cache
            print(f"Failed to load cache: {e}")
            return None
    
    def cache_is_valid(
        self,
        file_paths: List[str],
        config: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> bool:
        """
        Check if the cached index is still valid.
        
        Args:
            file_paths: List of current source file paths
            config: Current configuration
            metadata: Cached metadata
            
        Returns:
            True if cache is valid, False otherwise
        """
        # Check if files have changed
        current_hashes = {fp: compute_file_hash(fp) for fp in file_paths}
        cached_hashes = metadata.get("file_hashes", {})
        
        if current_hashes != cached_hashes:
            return False
        
        # Check if configuration has changed
        cached_config = metadata.get("config", {})
        
        # Check key configuration parameters
        key_params = [
            "chunk_size",
            "overlap",
            "embedding_model",
        ]
        
        for param in key_params:
            if config.get(param) != cached_config.get(param):
                return False
        
        return True
    
    def invalidate_cache(self, index_name: str) -> None:
        """
        Invalidate and remove cached index data.
        
        Args:
            index_name: Name/identifier for the index to invalidate
        """
        cache_dir = self.get_cache_path(index_name)
        
        # Remove all cache files
        for file in cache_dir.iterdir():
            if file.is_file():
                file.unlink()
        
        # Remove directory if empty
        if not any(cache_dir.iterdir()):
            cache_dir.rmdir()
    
    def clear_all_caches(self) -> None:
        """
        Clear all cached indexes.
        """
        for cache_subdir in self.cache_dir.iterdir():
            if cache_subdir.is_dir():
                for file in cache_subdir.iterdir():
                    if file.is_file():
                        file.unlink()
                cache_subdir.rmdir()


def get_default_cache() -> IndexCache:
    """
    Get the default index cache instance.
    
    Returns:
        IndexCache instance with default cache directory
    """
    return IndexCache()


__all__ = [
    'IndexCache',
    'get_default_cache',
]
