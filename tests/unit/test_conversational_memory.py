"""Unit tests for conversational (episodic) memory."""

import uuid

from src.memory.conversational import count_turns, get_history, save_turn


class TestSaveTurn:
    def test_save_turn_returns_persisted_object(self, db_session):
        thread_id = str(uuid.uuid4())
        turn = save_turn(db_session, thread_id, "user", "Hello")

        assert turn.id is not None
        assert turn.thread_id == thread_id
        assert turn.role == "user"
        assert turn.content == "Hello"
        assert turn.timestamp is not None

    def test_save_turn_with_metadata(self, db_session):
        thread_id = str(uuid.uuid4())
        turn = save_turn(db_session, thread_id, "assistant", "Hi", metadata={"source": "test"})

        assert turn.metadata_json == '{"source": "test"}'

    def test_save_turn_without_metadata(self, db_session):
        thread_id = str(uuid.uuid4())
        turn = save_turn(db_session, thread_id, "user", "Test")

        assert turn.metadata_json is None


class TestGetHistory:
    def test_get_history_returns_ordered_turns(self, db_session):
        thread_id = str(uuid.uuid4())
        save_turn(db_session, thread_id, "user", "First")
        save_turn(db_session, thread_id, "assistant", "Second")
        save_turn(db_session, thread_id, "user", "Third")

        history = get_history(db_session, thread_id)

        assert len(history) == 3
        assert history[0].content == "First"
        assert history[1].content == "Second"
        assert history[2].content == "Third"

    def test_get_history_respects_limit(self, db_session):
        thread_id = str(uuid.uuid4())
        for i in range(10):
            save_turn(db_session, thread_id, "user", f"Message {i}")

        history = get_history(db_session, thread_id, limit=3)

        assert len(history) == 3
        # Should return the 3 most recent, in chronological order
        assert history[0].content == "Message 7"
        assert history[2].content == "Message 9"

    def test_get_history_isolates_threads(self, db_session):
        thread_a = str(uuid.uuid4())
        thread_b = str(uuid.uuid4())
        save_turn(db_session, thread_a, "user", "Thread A")
        save_turn(db_session, thread_b, "user", "Thread B")

        history_a = get_history(db_session, thread_a)
        history_b = get_history(db_session, thread_b)

        assert len(history_a) == 1
        assert history_a[0].content == "Thread A"
        assert len(history_b) == 1
        assert history_b[0].content == "Thread B"

    def test_get_history_empty_thread(self, db_session):
        history = get_history(db_session, "nonexistent-thread")
        assert history == []


class TestCountTurns:
    def test_count_turns_returns_correct_count(self, db_session):
        thread_id = str(uuid.uuid4())
        save_turn(db_session, thread_id, "user", "One")
        save_turn(db_session, thread_id, "assistant", "Two")
        save_turn(db_session, thread_id, "user", "Three")

        assert count_turns(db_session, thread_id) == 3

    def test_count_turns_empty_thread(self, db_session):
        assert count_turns(db_session, "nonexistent") == 0
