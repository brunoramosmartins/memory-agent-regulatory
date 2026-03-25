"""Embedding generation for regulatory chunks and queries."""

from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_BATCH_SIZE = 32

_model: SentenceTransformer | None = None


def get_embedding_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    """Return embedding model instance (cached)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(model_name)
    return _model
