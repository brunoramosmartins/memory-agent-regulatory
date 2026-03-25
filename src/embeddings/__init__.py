"""Embeddings module - Vector representations for retrieval."""

from .embedding_generator import get_embedding_model
from .validation import (
    BGE_M3_DIMENSIONS,
    validate_chunk_embedding_pairs,
    validate_embedding,
    validate_embeddings_batch,
)

__all__ = [
    "get_embedding_model",
    "BGE_M3_DIMENSIONS",
    "validate_embedding",
    "validate_embeddings_batch",
    "validate_chunk_embedding_pairs",
]
