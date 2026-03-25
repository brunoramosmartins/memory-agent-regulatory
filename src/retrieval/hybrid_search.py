"""Hybrid search (BM25 + vector) over indexed regulatory chunks."""

import logging
from typing import Any

from weaviate.classes.query import HybridFusion

from src.vectorstore.weaviate_client import (
    BGE_M3_DIMENSIONS,
    CHUNK_COLLECTION,
    get_weaviate_client,
)

logger = logging.getLogger(__name__)

FUSION_TYPES = {
    "ranked": HybridFusion.RANKED,
    "relative_score": HybridFusion.RELATIVE_SCORE,
}


def hybrid_search(
    query: str,
    query_vector: list[float],
    top_k: int = 5,
    alpha: float = 0.5,
    fusion_type: str = "ranked",
) -> list[dict[str, Any]]:
    """
    Execute hybrid search (BM25 + vector) in Weaviate.

    Combines keyword (BM25) and semantic (vector) search using Weaviate's
    native hybrid query. The alpha parameter controls the balance:
    0.0 = pure BM25, 1.0 = pure vector, 0.5 = equal weight.

    Parameters
    ----------
    query : str
        The search query string (used for BM25 component).
    query_vector : list[float]
        Pre-computed query embedding (used for vector component).
    top_k : int
        Maximum number of results to return.
    alpha : float
        Balance between BM25 and vector search (0.0 to 1.0).
    fusion_type : str
        Score fusion strategy: "ranked" (RRF) or "relative_score".

    Returns
    -------
    list[dict[str, Any]]
        Results with same structure as vector_search for compatibility.
    """
    if len(query_vector) != BGE_M3_DIMENSIONS:
        raise ValueError(
            f"Invalid embedding dimension: expected {BGE_M3_DIMENSIONS}, "
            f"got {len(query_vector)}"
        )

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(
            f"alpha must be between 0.0 and 1.0, got {alpha}"
        )

    fusion_enum = FUSION_TYPES.get(fusion_type)
    if fusion_enum is None:
        raise ValueError(
            f"Invalid fusion_type: {fusion_type!r}. "
            f"Must be one of: {list(FUSION_TYPES.keys())}"
        )

    client = get_weaviate_client()
    collection = client.collections.get(CHUNK_COLLECTION)

    response = collection.query.hybrid(
        query=query,
        vector=query_vector,
        alpha=alpha,
        fusion_type=fusion_enum,
        limit=top_k,
        return_metadata=["score"],
    )

    results: list[dict[str, Any]] = []
    for obj in response.objects:
        props = obj.properties
        score = (
            getattr(obj.metadata, "score", None)
            if hasattr(obj, "metadata")
            else None
        )

        results.append(
            {
                "chunk_id": props.get("chunk_id"),
                "document_id": props.get("document_id"),
                "page_number": props.get("page_number"),
                "section_title": props.get("section_title"),
                "text": props.get("text"),
                "source_file": props.get("source_file"),
                "similarity_score": round(float(score), 4) if score is not None else None,
            }
        )

    return results
