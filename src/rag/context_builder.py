"""Context builder with real-tokenizer-based greedy packing."""

from typing import TYPE_CHECKING

from src.utils.tokenizer import count_tokens

if TYPE_CHECKING:
    from src.retrieval.models import RetrievalResult

SEPARATOR = "\n\n"


def _format_chunk(chunk: "RetrievalResult") -> str:
    """Format a single chunk with its source marker."""
    marker = f"[{chunk.document_id}, p. {chunk.page_number}]"
    return f"{marker}\n{chunk.text}"


def build_context(
    chunks: list["RetrievalResult"],
    max_chunks: int = 20,
    max_tokens: int | None = None,
) -> str:
    """
    Build context string by greedily packing chunks into a token budget.

    Chunks are added in order (highest similarity first) until the token
    budget is exhausted. No chunk is truncated mid-text; either the whole
    chunk fits or it is skipped.

    Parameters
    ----------
    chunks : list[RetrievalResult]
        Retrieved chunks, ordered by relevance (highest first).
    max_chunks : int
        Safety cap on number of chunks (default 20).
    max_tokens : int | None
        Token budget. When None, all chunks (up to max_chunks) are included.
    """
    if not chunks:
        return ""

    if max_tokens is None:
        limited = chunks[:max_chunks]
        parts = [_format_chunk(c) for c in limited]
        return SEPARATOR.join(parts)

    parts: list[str] = []
    total_tokens = 0

    for chunk in chunks[:max_chunks]:
        formatted = _format_chunk(chunk)
        chunk_tokens = count_tokens(formatted)

        # Account for separator between chunks
        sep_tokens = count_tokens(SEPARATOR) if parts else 0

        if total_tokens + sep_tokens + chunk_tokens > max_tokens:
            break

        parts.append(formatted)
        total_tokens += sep_tokens + chunk_tokens

    return SEPARATOR.join(parts)
