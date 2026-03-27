"""Conversational (episodic) memory — PostgreSQL-backed turn storage."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from src.memory.models import ConversationTurn

logger = logging.getLogger(__name__)


def save_turn(
    session: Session,
    thread_id: str,
    role: str,
    content: str,
    metadata: dict | None = None,
) -> ConversationTurn:
    """Persist a single conversation turn.

    Args:
        session: SQLAlchemy session.
        thread_id: Conversation thread identifier.
        role: Message role ("user", "assistant", "system").
        content: Message content.
        metadata: Optional JSON-serialisable metadata.

    Returns:
        The persisted ConversationTurn.
    """
    turn = ConversationTurn(
        thread_id=thread_id,
        role=role,
        content=content,
        timestamp=datetime.now(timezone.utc),
        metadata_json=json.dumps(metadata) if metadata else None,
    )
    session.add(turn)
    session.commit()
    logger.debug("Saved turn for thread=%s role=%s", thread_id, role)
    return turn


def get_history(
    session: Session,
    thread_id: str,
    limit: int = 20,
) -> list[ConversationTurn]:
    """Retrieve recent turns for a thread, ordered oldest-first.

    Args:
        session: SQLAlchemy session.
        thread_id: Conversation thread identifier.
        limit: Maximum number of turns to return.

    Returns:
        List of ConversationTurn ordered by timestamp ascending.
    """
    turns = (
        session.query(ConversationTurn)
        .filter(ConversationTurn.thread_id == thread_id)
        .order_by(ConversationTurn.timestamp.desc())
        .limit(limit)
        .all()
    )
    return list(reversed(turns))


def count_turns(session: Session, thread_id: str) -> int:
    """Count the number of turns in a thread."""
    return (
        session.query(ConversationTurn)
        .filter(ConversationTurn.thread_id == thread_id)
        .count()
    )
