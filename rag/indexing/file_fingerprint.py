"""
File Fingerprinting Module

This module provides SHA256 file fingerprinting for cache validation.
It allows the RAG system to detect when source files have changed
and invalidate cached indexes accordingly.
"""

import hashlib
from pathlib import Path
from typing import Optional


def compute_file_hash(file_path: str) -> str:
    """
    Compute SHA256 hash of a file for cache validation.
    
    This function reads the file in binary mode and computes its SHA256 hash,
    which serves as a unique fingerprint for the file's content.
    
    Args:
        file_path: Path to the file to hash
        
    Returns:
        Hexadecimal SHA256 hash string
        
    Raises:
        FileNotFoundError: If the file does not exist
        IOError: If the file cannot be read
    """
    sha256_hash = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        # Read file in chunks to handle large files efficiently
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    
    return sha256_hash.hexdigest()


def compute_multiple_file_hash(file_paths: list[str]) -> str:
    """
    Compute combined SHA256 hash of multiple files.
    
    This function computes hashes for multiple files and combines them
    into a single hash, useful when the cache depends on multiple source files.
    
    Args:
        file_paths: List of file paths to hash
        
    Returns:
        Hexadecimal SHA256 hash string representing the combined files
    """
    combined_hash = hashlib.sha256()
    
    # Sort paths to ensure consistent ordering
    sorted_paths = sorted(file_paths)
    
    for file_path in sorted_paths:
        file_hash = compute_file_hash(file_path)
        combined_hash.update(file_hash.encode('utf-8'))
    
    return combined_hash.hexdigest()


def file_has_changed(file_path: str, cached_hash: str) -> bool:
    """
    Check if a file has changed compared to a cached hash.
    
    Args:
        file_path: Path to the file to check
        cached_hash: Previously cached SHA256 hash
        
    Returns:
        True if the file has changed, False otherwise
    """
    try:
        current_hash = compute_file_hash(file_path)
        return current_hash != cached_hash
    except (FileNotFoundError, IOError):
        # If file cannot be read, treat as changed
        return True


def get_file_modification_time(file_path: str) -> Optional[float]:
    """
    Get the modification time of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Modification time as Unix timestamp, or None if file doesn't exist
    """
    try:
        path = Path(file_path)
        return path.stat().st_mtime
    except (FileNotFoundError, OSError):
        return None


__all__ = [
    'compute_file_hash',
    'compute_multiple_file_hash',
    'file_has_changed',
    'get_file_modification_time',
]
