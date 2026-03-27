"""MemoryManager — unified read/write facade for all memory types."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import weaviate
from sqlalchemy.orm import Session

from src.memory import conversational, procedural, semantic, summary
from src.memory.models import ConversationTurn
from src.memory.procedural import ProceduralPattern
from src.memory.semantic import SemanticResult

logger = logging.getLogger(__name__)


@dataclass
class MemoryContext:
    """Aggregated context from all memory sources."""

    history: list[ConversationTurn] = field(default_factory=list)
    semantic_results: list[SemanticResult] = field(default_factory=list)
    patterns: list[ProceduralPattern] = field(default_factory=list)
    summary: str | None = None


class MemoryManager:
    """Unified facade over conversational, semantic, procedural, and summary memory.

    The agent and Context Builder should interact with memory exclusively
    through this class, never directly with individual memory modules.
    """

    def __init__(
        self,
        db_session: Session,
        weaviate_client: weaviate.WeaviateClient,
        max_history_turns: int = 20,
        summary_threshold: int = 15,
        semantic_top_k: int = 5,
        procedural_top_k: int = 3,
    ) -> None:
        self._db = db_session
        self._weaviate = weaviate_client
        self._max_history_turns = max_history_turns
        self._summary_threshold = summary_threshold
        self._semantic_top_k = semantic_top_k
        self._procedural_top_k = procedural_top_k

    def read_context(
        self,
        thread_id: str,
        query_embedding: list[float],
    ) -> MemoryContext:
        """Read from all memory sources and return aggregated context.

        Args:
            thread_id: Conversation thread identifier.
            query_embedding: Pre-computed embedding of the current query.

        Returns:
            MemoryContext with history, semantic results, patterns, and summary.
        """
        start = time.monotonic()

        history = conversational.get_history(
            self._db, thread_id, limit=self._max_history_turns
        )

        semantic_results = semantic.search(
            self._weaviate, query_embedding, limit=self._semantic_top_k
        )

        patterns = procedural.find_patterns(
            self._weaviate, query_embedding, limit=self._procedural_top_k
        )

        summary_text = summary.get_summary(self._db, thread_id)

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info(
            "read_context thread=%s history=%d semantic=%d patterns=%d "
            "has_summary=%s latency_ms=%.1f",
            thread_id,
            len(history),
            len(semantic_results),
            len(patterns),
            summary_text is not None,
            elapsed_ms,
        )

        return MemoryContext(
            history=history,
            semantic_results=semantic_results,
            patterns=patterns,
            summary=summary_text,
        )

    def write_turn(
        self,
        thread_id: str,
        role: str,
        content: str,
        metadata: dict | None = None,
    ) -> ConversationTurn:
        """Write a conversation turn and trigger auto-summarization if needed.

        Args:
            thread_id: Conversation thread identifier.
            role: Message role ("user", "assistant").
            content: Message content.
            metadata: Optional metadata.

        Returns:
            The persisted ConversationTurn.
        """
        start = time.monotonic()

        turn = conversational.save_turn(self._db, thread_id, role, content, metadata)

        turn_count = conversational.count_turns(self._db, thread_id)
        if summary.should_summarize(turn_count, self._summary_threshold):
            logger.info(
                "Auto-summarization triggered thread=%s turn_count=%d",
                thread_id,
                turn_count,
            )
            self._auto_summarize(thread_id)

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.debug(
            "write_turn thread=%s role=%s latency_ms=%.1f",
            thread_id,
            role,
            elapsed_ms,
        )
        return turn

    def write_semantic(
        self,
        content: str,
        embedding: list[float],
        source: str,
        thread_id: str,
    ) -> str:
        """Write a semantic memory entry.

        Returns:
            The Weaviate object UUID.
        """
        start = time.monotonic()
        object_id = semantic.store(self._weaviate, content, embedding, source, thread_id)
        elapsed_ms = (time.monotonic() - start) * 1000
        logger.debug(
            "write_semantic thread=%s object_id=%s latency_ms=%.1f",
            thread_id,
            object_id,
            elapsed_ms,
        )
        return object_id

    def write_pattern(
        self,
        trigger: str,
        action: str,
        embedding: list[float],
        metadata: dict | None = None,
    ) -> str:
        """Write a procedural memory pattern.

        Returns:
            The Weaviate object UUID.
        """
        start = time.monotonic()
        object_id = procedural.store_pattern(
            self._weaviate, trigger, action, embedding, metadata
        )
        elapsed_ms = (time.monotonic() - start) * 1000
        logger.debug(
            "write_pattern object_id=%s latency_ms=%.1f",
            object_id,
            elapsed_ms,
        )
        return object_id

    def _auto_summarize(self, thread_id: str) -> None:
        """Compress conversation history into a summary.

        Uses a simple concatenation for now. Phase 3 will replace this
        with an LLM-based summarization call.
        """
        history = conversational.get_history(self._db, thread_id, limit=100)
        if not history:
            return

        lines = [f"{t.role}: {t.content}" for t in history]
        summary_text = (
            f"Conversation summary ({len(history)} turns):\n" + "\n".join(lines[-10:])
        )
        summary.save_summary(self._db, thread_id, summary_text)
        logger.info("Auto-summarized thread=%s turns=%d", thread_id, len(history))
