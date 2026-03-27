"""Tests for session generator."""

import json
import os
import tempfile

from src.simulation.session_generator import (
    TOPICS,
    generate_sessions,
    load_sessions,
    save_sessions,
)


class TestGenerateSessions:
    def test_generates_correct_count(self):
        sessions = generate_sessions(count_per_topic=5, seed=42)
        # 5 topics × 5 sessions each = 25
        assert len(sessions) == 25

    def test_generates_for_all_topics(self):
        sessions = generate_sessions(count_per_topic=3, seed=42)
        topics = {s.topic for s in sessions}
        assert topics == set(TOPICS.keys())

    def test_generates_for_specific_topics(self):
        sessions = generate_sessions(
            topics=["pix_fees", "pix_compliance"],
            count_per_topic=5,
            seed=42,
        )
        topics = {s.topic for s in sessions}
        assert topics == {"pix_fees", "pix_compliance"}
        assert len(sessions) == 10

    def test_reproducible_with_same_seed(self):
        s1 = generate_sessions(count_per_topic=3, seed=99)
        s2 = generate_sessions(count_per_topic=3, seed=99)
        assert len(s1) == len(s2)
        for a, b in zip(s1, s2, strict=False):
            assert a.session_id == b.session_id
            assert a.turn_count == b.turn_count

    def test_different_seeds_produce_different_sessions(self):
        s1 = generate_sessions(count_per_topic=3, seed=1)
        s2 = generate_sessions(count_per_topic=3, seed=2)
        # Session IDs differ because seed is part of the hash
        ids1 = {s.session_id for s in s1}
        ids2 = {s.session_id for s in s2}
        assert ids1 != ids2

    def test_sessions_have_turns_in_range(self):
        sessions = generate_sessions(
            count_per_topic=10,
            turns_range=(3, 6),
            seed=42,
        )
        for s in sessions:
            user_turns = len(s.user_turns)
            assert 3 <= user_turns <= 6, (
                f"Session {s.session_id} has {user_turns} user turns"
            )

    def test_sessions_have_implicit_references(self):
        sessions = generate_sessions(count_per_topic=10, seed=42)
        with_implicit = sum(1 for s in sessions if s.has_implicit_reference)
        # All sessions should have at least one implicit reference
        assert with_implicit == len(sessions)

    def test_unique_session_ids(self):
        sessions = generate_sessions(count_per_topic=10, seed=42)
        ids = [s.session_id for s in sessions]
        assert len(ids) == len(set(ids))

    def test_first_turn_is_user(self):
        sessions = generate_sessions(count_per_topic=5, seed=42)
        for s in sessions:
            assert s.turns[0].role == "user"

    def test_ignores_unknown_topic(self):
        sessions = generate_sessions(
            topics=["pix_fees", "nonexistent_topic"],
            count_per_topic=3,
            seed=42,
        )
        assert len(sessions) == 3
        assert all(s.topic == "pix_fees" for s in sessions)


class TestSaveLoadSessions:
    def test_save_and_load_roundtrip(self):
        sessions = generate_sessions(count_per_topic=2, seed=42)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sessions.json")
            save_sessions(sessions, path)

            loaded = load_sessions(path)
            assert len(loaded) == len(sessions)
            for orig, restored in zip(sessions, loaded, strict=False):
                assert orig.session_id == restored.session_id
                assert orig.topic == restored.topic
                assert orig.turn_count == restored.turn_count

    def test_saved_file_is_valid_json(self):
        sessions = generate_sessions(
            topics=["pix_fees"], count_per_topic=2, seed=42
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sessions.json")
            save_sessions(sessions, path)

            with open(path) as f:
                data = json.load(f)
            assert isinstance(data, list)
            assert len(data) == 2

    def test_creates_parent_directory(self):
        sessions = generate_sessions(
            topics=["pix_fees"], count_per_topic=1, seed=42
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nested", "dir", "sessions.json")
            save_sessions(sessions, path)
            assert os.path.exists(path)
