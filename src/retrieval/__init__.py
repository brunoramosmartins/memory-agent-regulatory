"""Retrieval module - Similarity search and document retrieval."""

from .models import RetrievalResult

__all__ = [
    "embed_query",
    "vector_search",
    "keyword_search",
    "hybrid_search",
    "retrieve",
    "RetrievalResult",
]


def __getattr__(name: str):
    """Lazy import for heavy modules (embed_query, retrieve, vector_search, etc.)."""
    if name == "embed_query":
        from .query_embedding import embed_query
        return embed_query
    if name == "retrieve":
        from .retriever import retrieve
        return retrieve
    if name == "vector_search":
        from .vector_search import vector_search
        return vector_search
    if name == "keyword_search":
        from .keyword_search import keyword_search
        return keyword_search
    if name == "hybrid_search":
        from .hybrid_search import hybrid_search
        return hybrid_search
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
