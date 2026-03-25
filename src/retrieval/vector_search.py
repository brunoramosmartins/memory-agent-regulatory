"""Semantic vector search over indexed regulatory chunks."""

from typing import Any

from src.vectorstore.weaviate_client import (
    BGE_M3_DIMENSIONS,
    CHUNK_COLLECTION,
    get_weaviate_client,
)


def vector_search(
    query_vector: list[float],
    top_k: int = 5,
    min_similarity: float = 0.0,
) -> list[dict[str, Any]]:
    """
    Execute vector similarity search in Weaviate.

    Receives pre-computed query vector. Does not perform query embedding.
    Returns list of results with metadata and similarity score.
    """
    if len(query_vector) != BGE_M3_DIMENSIONS:
        raise ValueError(
            f"Invalid embedding dimension: expected {BGE_M3_DIMENSIONS}, "
            f"got {len(query_vector)}"
        )
    client = get_weaviate_client()
    collection = client.collections.get(CHUNK_COLLECTION)

    response = collection.query.near_vector(
        near_vector=query_vector,
        limit=top_k,
        return_metadata=["distance"],
    )

    results: list[dict[str, Any]] = []
    for obj in response.objects:
        props = obj.properties
        # Weaviate returns distance (cosine: 0=identical). Convert to similarity (0-1).
        distance = (
            getattr(obj.metadata, "distance", None)
            if hasattr(obj, "metadata")
            else None
        )
        similarity = round(1 - float(distance), 4) if distance is not None else None

        results.append(
            {
                "chunk_id": props.get("chunk_id"),
                "document_id": props.get("document_id"),
                "page_number": props.get("page_number"),
                "section_title": props.get("section_title"),
                "text": props.get("text"),
                "source_file": props.get("source_file"),
                "similarity_score": similarity,
            }
        )

    if min_similarity > 0.0:
        results = [
            r for r in results
            if r["similarity_score"] is not None
            and r["similarity_score"] >= min_similarity
        ]

    return results
