"""Validation utilities for synthetic sessions."""

from __future__ import annotations

from src.simulation.models import Session, validate_session


def validate_sessions(
    sessions: list[Session],
    min_turns: int = 3,
    max_turns: int = 8,
    require_implicit_ref: bool = True,
) -> dict[str, list[str]]:
    """Validate all sessions and return errors keyed by session_id.

    Returns:
        Dict mapping session_id -> list of error messages.
        Only sessions with errors are included.
    """
    errors: dict[str, list[str]] = {}
    for session in sessions:
        session_errors = validate_session(
            session,
            min_turns=min_turns,
            max_turns=max_turns,
            require_implicit_ref=require_implicit_ref,
        )
        if session_errors:
            errors[session.session_id] = session_errors
    return errors


def session_stats(sessions: list[Session]) -> dict[str, int | float]:
    """Compute summary statistics for a list of sessions."""
    if not sessions:
        return {"total": 0}

    total_turns = sum(s.turn_count for s in sessions)
    user_turns = sum(len(s.user_turns) for s in sessions)
    topics = set(s.topic for s in sessions)
    implicit_count = sum(1 for s in sessions if s.has_implicit_reference)

    return {
        "total_sessions": len(sessions),
        "total_turns": total_turns,
        "total_user_turns": user_turns,
        "unique_topics": len(topics),
        "avg_turns_per_session": round(total_turns / len(sessions), 1),
        "sessions_with_implicit_ref": implicit_count,
        "implicit_ref_pct": round(100 * implicit_count / len(sessions), 1),
    }
