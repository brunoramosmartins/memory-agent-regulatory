"""Summary memory — compresses long conversation histories."""

from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from src.memory.models import SessionSummary

logger = logging.getLogger(__name__)


def get_summary(session: Session, thread_id: str) -> str | None:
    """Retrieve an existing session summary.

    Returns:
        The summary text, or None if no summary exists.
    """
    row = (
        session.query(SessionSummary)
        .filter(SessionSummary.thread_id == thread_id)
        .first()
    )
    return row.summary if row else None


def save_summary(session: Session, thread_id: str, summary_text: str) -> SessionSummary:
    """Save or update a session summary.

    If a summary already exists for the thread, it is replaced.
    """
    existing = (
        session.query(SessionSummary)
        .filter(SessionSummary.thread_id == thread_id)
        .first()
    )
    if existing:
        existing.summary = summary_text
        session.commit()
        logger.debug("Updated summary for thread=%s", thread_id)
        return existing

    row = SessionSummary(thread_id=thread_id, summary=summary_text)
    session.add(row)
    session.commit()
    logger.debug("Created summary for thread=%s", thread_id)
    return row


def should_summarize(turn_count: int, threshold: int = 15) -> bool:
    """Check whether a thread should be summarized based on turn count.

    Args:
        turn_count: Current number of turns in the thread.
        threshold: Number of turns that triggers summarization.

    Returns:
        True if turn_count >= threshold.
    """
    return turn_count >= threshold
