"""Context builder — merges memory context and retrieval into a single LLM prompt.

Priority order (highest → lowest):
    1. Summary (compressed history — highest info density)
    2. Conversational history (recent turns)
    3. Semantic memory (cross-session knowledge)
    4. Procedural patterns (workflow hints)
    5. Retrieval results (document chunks)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.memory.manager import MemoryContext

if TYPE_CHECKING:
    from src.retrieval.models import RetrievalResult

SEPARATOR = "\n\n"

# Character-based token approximation (4 chars ≈ 1 token).
# Avoids requiring the heavy tokenizer model at build time.
_CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    """Estimate token count from character length."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _format_chunk(chunk: RetrievalResult) -> str:
    """Format a single retrieval chunk with its source marker."""
    marker = f"[{chunk.document_id}, p. {chunk.page_number}]"
    return f"{marker}\n{chunk.text}"


def _build_section(header: str, content: str) -> str:
    """Wrap content with a section header."""
    return f"### {header}\n{content}"


def _format_history(history: list) -> str:
    """Format conversation history turns."""
    if not history:
        return ""
    lines = [f"{turn.role.capitalize()}: {turn.content}" for turn in history]
    return _build_section("Conversation History", "\n".join(lines))


def _format_semantic(results: list) -> str:
    """Format semantic memory results."""
    if not results:
        return ""
    lines = [f"- {r.content} (score: {r.score:.2f})" for r in results]
    return _build_section("Related Knowledge", "\n".join(lines))


def _format_patterns(patterns: list) -> str:
    """Format procedural patterns."""
    if not patterns:
        return ""
    lines = [f"- When: {p.trigger} → Then: {p.action}" for p in patterns]
    return _build_section("Workflow Patterns", "\n".join(lines))


def _format_summary(summary_text: str | None) -> str:
    """Format session summary."""
    if not summary_text:
        return ""
    return _build_section("Session Summary", summary_text)


def _format_retrieval(chunks: list[RetrievalResult]) -> str:
    """Format retrieval chunks."""
    if not chunks:
        return ""
    parts = [_format_chunk(c) for c in chunks]
    return _build_section("Retrieved Documents", SEPARATOR.join(parts))


def build_context(
    chunks: list[RetrievalResult] | None = None,
    memory_context: MemoryContext | None = None,
    max_tokens: int | None = None,
) -> str:
    """Build context string by assembling memory and retrieval sections.

    Sections are added in priority order. If a token budget is set,
    lower-priority sections are dropped first.

    Parameters
    ----------
    chunks : list[RetrievalResult] | None
        Retrieved document chunks, ordered by relevance.
    memory_context : MemoryContext | None
        Aggregated memory from MemoryManager.read_context().
    max_tokens : int | None
        Token budget. When None, all sections are included.
    """
    # Build sections in priority order (highest first)
    sections: list[str] = []

    if memory_context is not None:
        s = _format_summary(memory_context.summary)
        if s:
            sections.append(s)

        h = _format_history(memory_context.history)
        if h:
            sections.append(h)

        sem = _format_semantic(memory_context.semantic_results)
        if sem:
            sections.append(sem)

        pat = _format_patterns(memory_context.patterns)
        if pat:
            sections.append(pat)

    if chunks:
        ret = _format_retrieval(chunks)
        if ret:
            sections.append(ret)

    if not sections:
        return ""

    if max_tokens is None:
        return SEPARATOR.join(sections)

    # Greedy packing: keep adding sections while within budget.
    # Drop from the end (lowest priority) if over budget.
    result_parts: list[str] = []
    total_tokens = 0

    for section in sections:
        section_tokens = _estimate_tokens(section)
        sep_tokens = _estimate_tokens(SEPARATOR) if result_parts else 0

        if total_tokens + sep_tokens + section_tokens > max_tokens:
            continue

        result_parts.append(section)
        total_tokens += sep_tokens + section_tokens

    return SEPARATOR.join(result_parts)
