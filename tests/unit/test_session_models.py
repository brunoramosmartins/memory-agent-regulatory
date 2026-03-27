"""Tests for simulation data models."""

import json

from src.simulation.models import Session, Turn, validate_session

# ── Turn tests ───────────────────────────────────────────────────────────────


class TestTurn:
    def test_create_turn(self):
        turn = Turn(role="user", content="What is PIX?")
        assert turn.role == "user"
        assert turn.content == "What is PIX?"
        assert turn.expected_behavior == ""
        assert turn.metadata == {}

    def test_turn_with_behavior(self):
        turn = Turn(
            role="user",
            content="And what about that?",
            expected_behavior="implicit_reference",
        )
        assert turn.expected_behavior == "implicit_reference"

    def test_turn_with_metadata(self):
        turn = Turn(role="assistant", content="resp", metadata={"score": 0.9})
        assert turn.metadata["score"] == 0.9


# ── Session tests ────────────────────────────────────────────────────────────


class TestSession:
    def _make_session(self, num_user_turns=4, has_implicit=True):
        turns = []
        for i in range(num_user_turns):
            behavior = "implicit_reference" if (i == 1 and has_implicit) else "direct"
            turns.append(Turn(role="user", content=f"Q{i}", expected_behavior=behavior))
            turns.append(Turn(role="assistant", content=f"A{i}", expected_behavior="answer"))
        return Session(session_id="test-001", topic="pix_fees", turns=turns)

    def test_turn_count(self):
        s = self._make_session(num_user_turns=4)
        assert s.turn_count == 8  # 4 user + 4 assistant

    def test_user_turns(self):
        s = self._make_session(num_user_turns=3)
        assert len(s.user_turns) == 3

    def test_has_implicit_reference(self):
        s = self._make_session(has_implicit=True)
        assert s.has_implicit_reference is True

    def test_no_implicit_reference(self):
        s = self._make_session(has_implicit=False)
        assert s.has_implicit_reference is False

    def test_to_dict_roundtrip(self):
        s = self._make_session()
        d = s.to_dict()
        restored = Session.from_dict(d)
        assert restored.session_id == s.session_id
        assert restored.topic == s.topic
        assert restored.turn_count == s.turn_count

    def test_json_roundtrip(self):
        s = self._make_session()
        j = s.to_json()
        parsed = json.loads(j)
        assert parsed["session_id"] == "test-001"
        restored = Session.from_json(j)
        assert restored.turn_count == s.turn_count

    def test_metadata(self):
        s = Session(session_id="m1", topic="t", metadata={"seed": 42})
        assert s.metadata["seed"] == 42


# ── Validation tests ─────────────────────────────────────────────────────────


class TestValidation:
    def _make_session(self, num_user_turns=4, has_implicit=True, topic="pix_fees"):
        turns = []
        for i in range(num_user_turns):
            behavior = "implicit_reference" if (i == 1 and has_implicit) else "direct"
            turns.append(Turn(role="user", content=f"Q{i}", expected_behavior=behavior))
            turns.append(Turn(role="assistant", content=f"A{i}"))
        return Session(session_id="val-001", topic=topic, turns=turns)

    def test_valid_session(self):
        s = self._make_session(num_user_turns=4, has_implicit=True)
        errors = validate_session(s)
        assert errors == []

    def test_too_few_turns(self):
        s = self._make_session(num_user_turns=1)
        errors = validate_session(s, min_turns=3)
        assert any("Too few user turns" in e for e in errors)

    def test_too_many_turns(self):
        s = self._make_session(num_user_turns=10)
        errors = validate_session(s, max_turns=8)
        assert any("Too many user turns" in e for e in errors)

    def test_missing_implicit_ref(self):
        s = self._make_session(has_implicit=False)
        errors = validate_session(s, require_implicit_ref=True)
        assert any("implicit" in e.lower() for e in errors)

    def test_no_implicit_ref_ok_when_not_required(self):
        s = self._make_session(has_implicit=False)
        errors = validate_session(s, require_implicit_ref=False)
        assert errors == []

    def test_missing_session_id(self):
        s = Session(session_id="", topic="t", turns=[
            Turn(role="user", content="q", expected_behavior="implicit_reference"),
            Turn(role="assistant", content="a"),
            Turn(role="user", content="q2", expected_behavior="direct"),
            Turn(role="assistant", content="a2"),
            Turn(role="user", content="q3", expected_behavior="direct"),
            Turn(role="assistant", content="a3"),
        ])
        errors = validate_session(s)
        assert any("session_id" in e for e in errors)

    def test_missing_topic(self):
        s = Session(session_id="x", topic="", turns=[
            Turn(role="user", content="q", expected_behavior="implicit_reference"),
            Turn(role="assistant", content="a"),
            Turn(role="user", content="q2", expected_behavior="direct"),
            Turn(role="assistant", content="a2"),
            Turn(role="user", content="q3", expected_behavior="direct"),
            Turn(role="assistant", content="a3"),
        ])
        errors = validate_session(s)
        assert any("topic" in e.lower() for e in errors)
