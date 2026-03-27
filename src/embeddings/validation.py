"""Embedding validation for dimensionality and consistency."""

import logging
from typing import Any

logger = logging.getLogger(__name__)

BGE_M3_DIMENSIONS = 1024


def validate_embedding(
    embedding: list[float] | Any,
    expected_dim: int = BGE_M3_DIMENSIONS,
) -> None:
    """
    Validate a single embedding vector.

    Raises ValueError if validation fails.
    """
    if embedding is None:
        raise ValueError("Embedding is None")

    try:
        vec = list(embedding)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Embedding must be iterable of numbers: {e}") from e

    if len(vec) != expected_dim:
        raise ValueError(
            f"Embedding dimension mismatch: expected {expected_dim}, got {len(vec)}"
        )

    for i, x in enumerate(vec):
        if not isinstance(x, int | float):
            raise ValueError(
                f"Embedding element at index {i} is not numeric: {type(x)}"
            )
        if x != x:  # NaN check
            raise ValueError(f"Embedding contains NaN at index {i}")

    if len(vec) == 0:
        raise ValueError("Embedding is empty")


def validate_embeddings_batch(
    embeddings: list[list[float]],
    expected_dim: int = BGE_M3_DIMENSIONS,
) -> None:
    """
    Validate a batch of embeddings.

    Raises ValueError on first failure. Logs clear error messages.
    """
    for i, emb in enumerate(embeddings):
        try:
            validate_embedding(emb, expected_dim)
        except ValueError as e:
            logger.error("Embedding validation failed at index %d: %s", i, e)
            raise


def validate_chunk_embedding_pairs(
    pairs: list[tuple[Any, list[float]]],
    expected_dim: int = BGE_M3_DIMENSIONS,
) -> None:
    """
    Validate that each chunk has a valid embedding.

    Raises ValueError if any chunk lacks embedding or embedding is invalid.
    """
    for i, (chunk, embedding) in enumerate(pairs):
        if embedding is None:
            chunk_id = getattr(chunk, "chunk_id", f"index_{i}")
            raise ValueError(f"Chunk {chunk_id} has no embedding")
        validate_embedding(embedding, expected_dim)
