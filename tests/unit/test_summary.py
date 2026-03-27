"""Unit tests for summary memory."""

import uuid

from src.memory.summary import get_summary, save_summary, should_summarize


class TestSaveSummary:
    def test_save_new_summary(self, db_session):
        thread_id = str(uuid.uuid4())
        row = save_summary(db_session, thread_id, "User asked about fees.")

        assert row.thread_id == thread_id
        assert row.summary == "User asked about fees."
        assert row.created_at is not None

    def test_save_summary_updates_existing(self, db_session):
        thread_id = str(uuid.uuid4())
        save_summary(db_session, thread_id, "Original summary")
        save_summary(db_session, thread_id, "Updated summary")

        result = get_summary(db_session, thread_id)
        assert result == "Updated summary"


class TestGetSummary:
    def test_get_existing_summary(self, db_session):
        thread_id = str(uuid.uuid4())
        save_summary(db_session, thread_id, "Some summary")

        result = get_summary(db_session, thread_id)
        assert result == "Some summary"

    def test_get_nonexistent_summary(self, db_session):
        result = get_summary(db_session, "no-such-thread")
        assert result is None


class TestShouldSummarize:
    def test_below_threshold(self):
        assert should_summarize(10, threshold=15) is False

    def test_at_threshold(self):
        assert should_summarize(15, threshold=15) is True

    def test_above_threshold(self):
        assert should_summarize(20, threshold=15) is True

    def test_zero_turns(self):
        assert should_summarize(0, threshold=15) is False

    def test_custom_threshold(self):
        assert should_summarize(5, threshold=5) is True
        assert should_summarize(4, threshold=5) is False
