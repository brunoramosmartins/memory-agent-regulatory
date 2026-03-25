"""Vector store module - Weaviate integration for chunk indexing."""

from .weaviate_client import (
    get_weaviate_client,
    init_chunk_collection,
    is_weaviate_ready,
    validate_chunk_schema,
)

__all__ = [
    "get_weaviate_client",
    "init_chunk_collection",
    "is_weaviate_ready",
    "validate_chunk_schema",
]
