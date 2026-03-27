"""Procedural memory — stores recurring workflow patterns via Weaviate vectors."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import weaviate
from weaviate.classes.config import Configure, DataType, Property

logger = logging.getLogger(__name__)

PROCEDURAL_COLLECTION = "ProceduralMemory"


@dataclass
class ProceduralPattern:
    """A single procedural memory pattern."""

    object_id: str
    trigger: str
    action: str
    score: float


def init_procedural_collection(client: weaviate.WeaviateClient, recreate: bool = False) -> None:
    """Create the ProceduralMemory collection in Weaviate."""
    if recreate and client.collections.exists(PROCEDURAL_COLLECTION):
        client.collections.delete(PROCEDURAL_COLLECTION)

    if not client.collections.exists(PROCEDURAL_COLLECTION):
        client.collections.create(
            name=PROCEDURAL_COLLECTION,
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="trigger", data_type=DataType.TEXT),
                Property(name="action", data_type=DataType.TEXT),
                Property(name="metadata_json", data_type=DataType.TEXT),
                Property(name="timestamp", data_type=DataType.TEXT),
            ],
        )
        logger.info("Created collection %s", PROCEDURAL_COLLECTION)


def store_pattern(
    client: weaviate.WeaviateClient,
    trigger: str,
    action: str,
    embedding: list[float],
    metadata: dict | None = None,
) -> str:
    """Store a procedural pattern.

    Args:
        client: Weaviate client.
        trigger: Condition that activates this pattern.
        action: Action to take when triggered.
        embedding: Pre-computed embedding of the trigger text.
        metadata: Optional JSON-serialisable metadata.

    Returns:
        The Weaviate object UUID as a string.
    """
    import json

    collection = client.collections.get(PROCEDURAL_COLLECTION)
    object_id = str(uuid.uuid4())
    collection.data.insert(
        uuid=object_id,
        properties={
            "trigger": trigger,
            "action": action,
            "metadata_json": json.dumps(metadata) if metadata else "{}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        vector=embedding,
    )
    logger.debug("Stored procedural pattern object_id=%s", object_id)
    return object_id


def find_patterns(
    client: weaviate.WeaviateClient,
    query_embedding: list[float],
    limit: int = 3,
) -> list[ProceduralPattern]:
    """Find procedural patterns matching a query by vector similarity.

    Args:
        client: Weaviate client.
        query_embedding: Query embedding vector.
        limit: Maximum number of results.

    Returns:
        List of ProceduralPattern ordered by similarity (highest first).
    """
    collection = client.collections.get(PROCEDURAL_COLLECTION)

    response = collection.query.near_vector(
        near_vector=query_embedding,
        limit=limit,
        return_metadata=weaviate.classes.query.MetadataQuery(distance=True),
    )

    results = []
    for obj in response.objects:
        score = 1.0 - (obj.metadata.distance or 0.0)
        results.append(
            ProceduralPattern(
                object_id=str(obj.uuid),
                trigger=obj.properties.get("trigger", ""),
                action=obj.properties.get("action", ""),
                score=score,
            )
        )
    return results
