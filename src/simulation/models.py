"""Data models for synthetic session generation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Turn:
    """A single turn in a simulated conversation."""

    role: str  # "user" or "assistant"
    content: str
    expected_behavior: str = ""  # e.g. "implicit_reference", "follow_up", "direct"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    """A multi-turn simulated conversation session."""

    session_id: str
    topic: str
    turns: list[Turn] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    @property
    def user_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.role == "user"]

    @property
    def has_implicit_reference(self) -> bool:
        """Check if session contains at least one implicit reference turn."""
        return any(
            t.expected_behavior == "implicit_reference" for t in self.turns
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Session:
        turns = [Turn(**t) for t in data.get("turns", [])]
        return cls(
            session_id=data["session_id"],
            topic=data["topic"],
            turns=turns,
            metadata=data.get("metadata", {}),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, text: str) -> Session:
        return cls.from_dict(json.loads(text))


def validate_session(
    session: Session,
    min_turns: int = 3,
    max_turns: int = 8,
    require_implicit_ref: bool = True,
) -> list[str]:
    """Validate a session and return list of error messages (empty = valid)."""
    errors: list[str] = []

    user_turn_count = len(session.user_turns)
    if user_turn_count < min_turns:
        errors.append(
            f"Too few user turns: {user_turn_count} < {min_turns}"
        )
    if user_turn_count > max_turns:
        errors.append(
            f"Too many user turns: {user_turn_count} > {max_turns}"
        )

    if not session.session_id:
        errors.append("Missing session_id")

    if not session.topic:
        errors.append("Missing topic")

    if require_implicit_ref and not session.has_implicit_reference:
        errors.append("No implicit reference turn found")

    # Check alternating roles (user starts)
    user_turns = [t for t in session.turns if t.role == "user"]
    if len(user_turns) == 0:
        errors.append("No user turns found")

    return errors
