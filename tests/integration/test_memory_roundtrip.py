"""Integration tests for the memory layer.

These tests require running PostgreSQL and Weaviate instances.
Run with: pytest -m integration tests/integration/test_memory_roundtrip.py -v
"""

import uuid

import pytest

from src.memory.conversational import count_turns, get_history, save_turn
from src.memory.models import Base
from src.memory.summary import get_summary, save_summary, should_summarize

pytestmark = pytest.mark.integration


@pytest.fixture
def pg_session():
    """Create a real PostgreSQL session for integration tests."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from src.config import get_settings

    settings = get_settings()
    engine = create_engine(settings.database.url, echo=False)
    Base.metadata.create_all(engine)

    factory = sessionmaker(bind=engine, expire_on_commit=False)
    session = factory()
    yield session
    session.close()
    # Clean up test data
    Base.metadata.drop_all(engine)
    engine.dispose()


class TestConversationalRoundtrip:
    def test_save_and_retrieve_turns(self, pg_session):
        thread_id = str(uuid.uuid4())

        save_turn(pg_session, thread_id, "user", "What are PIX fees?")
        save_turn(pg_session, thread_id, "assistant", "PIX has no fees for individuals.")
        save_turn(pg_session, thread_id, "user", "What about businesses?")

        history = get_history(pg_session, thread_id, limit=10)
        assert len(history) == 3
        assert history[0].role == "user"
        assert history[1].role == "assistant"
        assert count_turns(pg_session, thread_id) == 3

    def test_thread_isolation(self, pg_session):
        thread_a = str(uuid.uuid4())
        thread_b = str(uuid.uuid4())

        save_turn(pg_session, thread_a, "user", "Question A")
        save_turn(pg_session, thread_b, "user", "Question B")

        assert len(get_history(pg_session, thread_a)) == 1
        assert len(get_history(pg_session, thread_b)) == 1


class TestSummaryRoundtrip:
    def test_save_and_get_summary(self, pg_session):
        thread_id = str(uuid.uuid4())
        save_summary(pg_session, thread_id, "User discussed fees and deadlines.")

        result = get_summary(pg_session, thread_id)
        assert result == "User discussed fees and deadlines."

    def test_summary_update(self, pg_session):
        thread_id = str(uuid.uuid4())
        save_summary(pg_session, thread_id, "V1")
        save_summary(pg_session, thread_id, "V2")

        assert get_summary(pg_session, thread_id) == "V2"

    def test_should_summarize_integration(self, pg_session):
        thread_id = str(uuid.uuid4())
        for i in range(16):
            save_turn(pg_session, thread_id, "user", f"Turn {i}")

        count = count_turns(pg_session, thread_id)
        assert should_summarize(count, threshold=15) is True
