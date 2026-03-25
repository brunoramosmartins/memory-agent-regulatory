"""Cross-encoder reranking for retrieval results."""

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import RetrievalResult

logger = logging.getLogger(__name__)

_cross_encoder = None


def _get_cross_encoder(model_name: str):
    """Load and cache cross-encoder model. Returns None on failure."""
    global _cross_encoder
    if _cross_encoder is not None:
        return _cross_encoder
    try:
        from sentence_transformers import CrossEncoder

        _cross_encoder = CrossEncoder(model_name)
        logger.info("Loaded cross-encoder model: %s", model_name)
        return _cross_encoder
    except Exception as e:
        logger.warning("Failed to load cross-encoder '%s': %s", model_name, e)
        return None


def rerank(
    query: str,
    chunks: list["RetrievalResult"],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n: int = 5,
) -> list["RetrievalResult"]:
    """
    Rerank chunks using a cross-encoder model.

    Graceful degradation: if cross-encoder is unavailable or prediction
    fails, returns chunks[:top_n] with original ordering.

    Parameters
    ----------
    query : str
        The user query.
    chunks : list[RetrievalResult]
        Chunks from initial retrieval (ordered by vector similarity).
    model_name : str
        HuggingFace model identifier for the cross-encoder.
    top_n : int
        Number of top results to return after reranking.

    Returns
    -------
    list[RetrievalResult]
        Reranked chunks, ordered by cross-encoder score (highest first).
    """
    if not chunks:
        return []

    encoder = _get_cross_encoder(model_name)
    if encoder is None:
        logger.info(
            "Cross-encoder unavailable; returning top-%d by vector similarity.", top_n,
        )
        return chunks[:top_n]

    pairs = [(query, chunk.text) for chunk in chunks]

    try:
        scores = encoder.predict(pairs)
    except Exception as e:
        logger.warning("Cross-encoder prediction failed: %s. Falling back.", e)
        return chunks[:top_n]

    scored = sorted(
        zip(chunks, scores, strict=True), key=lambda x: float(x[1]), reverse=True,
    )

    reranked = []
    for chunk, score in scored[:top_n]:
        reranked.append(
            replace(chunk, similarity_score=round(float(score), 4))
        )

    return reranked
