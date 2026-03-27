"""Semantic memory — Weaviate vector-backed storage for cross-thread knowledge."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import weaviate
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import Filter

logger = logging.getLogger(__name__)

SEMANTIC_COLLECTION = "SemanticMemory"
BGE_M3_DIMENSIONS = 1024


@dataclass
class SemanticResult:
    """A single semantic memory search result."""

    object_id: str
    content: str
    source: str
    thread_id: str
    score: float


def init_semantic_collection(client: weaviate.WeaviateClient, recreate: bool = False) -> None:
    """Create the SemanticMemory collection in Weaviate."""
    if recreate and client.collections.exists(SEMANTIC_COLLECTION):
        client.collections.delete(SEMANTIC_COLLECTION)

    if not client.collections.exists(SEMANTIC_COLLECTION):
        client.collections.create(
            name=SEMANTIC_COLLECTION,
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="content", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="thread_id", data_type=DataType.TEXT),
                Property(name="timestamp", data_type=DataType.TEXT),
            ],
        )
        logger.info("Created collection %s", SEMANTIC_COLLECTION)


def store(
    client: weaviate.WeaviateClient,
    content: str,
    embedding: list[float],
    source: str,
    thread_id: str,
) -> str:
    """Store a semantic memory entry.

    Args:
        client: Weaviate client.
        content: Text content to store.
        embedding: Pre-computed embedding vector.
        source: Origin of the content (e.g. "agent_response", "retrieval").
        thread_id: Conversation thread identifier.

    Returns:
        The Weaviate object UUID as a string.
    """
    collection = client.collections.get(SEMANTIC_COLLECTION)
    object_id = str(uuid.uuid4())
    collection.data.insert(
        uuid=object_id,
        properties={
            "content": content,
            "source": source,
            "thread_id": thread_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        vector=embedding,
    )
    logger.debug("Stored semantic memory object_id=%s thread=%s", object_id, thread_id)
    return object_id


def search(
    client: weaviate.WeaviateClient,
    query_embedding: list[float],
    limit: int = 5,
    thread_id: str | None = None,
) -> list[SemanticResult]:
    """Search semantic memory by vector similarity.

    Args:
        client: Weaviate client.
        query_embedding: Query embedding vector.
        limit: Maximum number of results.
        thread_id: Optional filter to restrict results to a specific thread.

    Returns:
        List of SemanticResult ordered by similarity (highest first).
    """
    collection = client.collections.get(SEMANTIC_COLLECTION)

    filters = None
    if thread_id is not None:
        filters = Filter.by_property("thread_id").equal(thread_id)

    response = collection.query.near_vector(
        near_vector=query_embedding,
        limit=limit,
        filters=filters,
        return_metadata=weaviate.classes.query.MetadataQuery(distance=True),
    )

    results = []
    for obj in response.objects:
        score = 1.0 - (obj.metadata.distance or 0.0)
        results.append(
            SemanticResult(
                object_id=str(obj.uuid),
                content=obj.properties.get("content", ""),
                source=obj.properties.get("source", ""),
                thread_id=obj.properties.get("thread_id", ""),
                score=score,
            )
        )
    return results


def delete(client: weaviate.WeaviateClient, object_id: str) -> bool:
    """Delete a semantic memory entry by UUID.

    Returns:
        True if deletion was successful.
    """
    collection = client.collections.get(SEMANTIC_COLLECTION)
    try:
        collection.data.delete_by_id(object_id)
        logger.debug("Deleted semantic memory object_id=%s", object_id)
        return True
    except Exception:
        logger.warning("Failed to delete semantic memory object_id=%s", object_id)
        return False
