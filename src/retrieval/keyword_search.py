"""BM25 keyword search over indexed regulatory chunks."""

import logging
from typing import Any

from src.vectorstore.weaviate_client import CHUNK_COLLECTION, get_weaviate_client

logger = logging.getLogger(__name__)

DEFAULT_QUERY_PROPERTIES = ["text"]


def keyword_search(
    query: str,
    top_k: int = 5,
    query_properties: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Execute BM25 keyword search in Weaviate.

    Uses Weaviate's native BM25 index (enabled by default on TEXT properties).
    Returns list of results with metadata and BM25 score.

    Parameters
    ----------
    query : str
        The search query string for BM25 matching.
    top_k : int
        Maximum number of results to return.
    query_properties : list[str] | None
        Properties to search over. Defaults to ["text"].

    Returns
    -------
    list[dict[str, Any]]
        Results with same structure as vector_search for compatibility.
    """
    if query_properties is None:
        query_properties = DEFAULT_QUERY_PROPERTIES

    client = get_weaviate_client()
    collection = client.collections.get(CHUNK_COLLECTION)

    response = collection.query.bm25(
        query=query,
        limit=top_k,
        query_properties=query_properties,
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
