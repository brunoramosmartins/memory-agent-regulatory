"""Weaviate vector database client and schema configuration."""

import logging
from typing import Any

import weaviate
from weaviate.classes.config import Configure, DataType, Property

logger = logging.getLogger(__name__)

CHUNK_COLLECTION = "Chunk"
BGE_M3_DIMENSIONS = 1024

_client: weaviate.WeaviateClient | None = None


def get_weaviate_client(
    host: str | None = None,
    port: int | None = None,
    grpc_port: int | None = None,
) -> weaviate.WeaviateClient:
    """Connect to local Weaviate instance. Reuses connection if available.

    When called without arguments, reads host/port from centralized settings.
    """
    global _client
    if _client is None:
        if host is None or port is None or grpc_port is None:
            from src.config import get_settings

            ws = get_settings().weaviate
            host = host or ws.host
            port = port or ws.port
            grpc_port = grpc_port or ws.grpc_port
        _client = weaviate.connect_to_local(host=host, port=port, grpc_port=grpc_port)
    return _client


def is_weaviate_ready(host: str | None = None, port: int | None = None) -> bool:
    """Check if Weaviate is reachable via REST API."""
    import requests

    if host is None or port is None:
        from src.config import get_settings

        ws = get_settings().weaviate
        host = host or ws.host
        port = port or ws.port

    try:
        r = requests.get(f"http://{host}:{port}/v1/meta", timeout=2)
        return r.status_code == 200
    except Exception as e:
        logger.debug("Weaviate not reachable: %s", e)
        return False


def init_chunk_collection(
    client: weaviate.WeaviateClient | None = None,
    recreate: bool = False,
) -> None:
    """
    Create or recreate the Chunk collection with schema.

    Schema supports metadata filtering and 1024-dim vectors (BGE-M3).
    """
    if client is None:
        client = get_weaviate_client()

    if recreate and client.collections.exists(CHUNK_COLLECTION):
        client.collections.delete(CHUNK_COLLECTION)
        logger.info("Deleted existing collection %s", CHUNK_COLLECTION)

    if not client.collections.exists(CHUNK_COLLECTION):
        client.collections.create(
            name=CHUNK_COLLECTION,
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="chunk_id", data_type=DataType.TEXT),
                Property(name="document_id", data_type=DataType.TEXT),
                Property(name="page_number", data_type=DataType.INT),
                Property(name="segment_index", data_type=DataType.INT),
                Property(name="chunk_index", data_type=DataType.INT),
                Property(name="section_title", data_type=DataType.TEXT),
                Property(name="article_numbers", data_type=DataType.TEXT_ARRAY),
                Property(name="source_file", data_type=DataType.TEXT),
                Property(name="text", data_type=DataType.TEXT),
            ],
        )
        logger.info(
            "Created collection %s with %d-dim vectors",
            CHUNK_COLLECTION,
            BGE_M3_DIMENSIONS,
        )
    else:
        logger.info("Collection %s already exists", CHUNK_COLLECTION)


def validate_chunk_schema(client: weaviate.WeaviateClient | None = None) -> bool:
    """
    Validate that Chunk collection exists and is ready for indexing.

    Returns True if valid. Logs warnings and returns False otherwise.
    """
    if client is None:
        client = get_weaviate_client()

    if not client.collections.exists(CHUNK_COLLECTION):
        logger.warning("Collection %s does not exist", CHUNK_COLLECTION)
        return False

    logger.info("Schema validation passed: %s exists", CHUNK_COLLECTION)
    return True


def chunk_to_weaviate_properties(chunk: Any) -> dict[str, Any]:
    """Convert Chunk to Weaviate properties dict."""
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "page_number": chunk.page_number,
        "segment_index": chunk.segment_index,
        "chunk_index": chunk.chunk_index,
        "section_title": chunk.section_title or "",
        "article_numbers": chunk.article_numbers or [],
        "source_file": chunk.source_file,
        "text": chunk.text,
    }
