"""Unit tests for MemoryManager unified API."""

import uuid
from unittest.mock import MagicMock

import pytest

from src.memory.manager import MemoryContext, MemoryManager
from src.memory.models import ConversationTurn
from tests.helpers import make_test_session


def _make_fake_embedding(dims: int = 1024) -> list[float]:
    return [0.1] * dims


@pytest.fixture
def db_session():
    session = make_test_session()
    yield session
    session.close()


@pytest.fixture
def mock_weaviate():
    return MagicMock()


@pytest.fixture
def manager(db_session, mock_weaviate):
    return MemoryManager(
        db_session=db_session,
        weaviate_client=mock_weaviate,
        max_history_turns=20,
        summary_threshold=5,
        semantic_top_k=5,
        procedural_top_k=3,
    )


class TestReadContext:
    def test_read_empty_context(self, manager, mock_weaviate):
        # Mock Weaviate to return empty results
        collection_mock = MagicMock()
        mock_weaviate.collections.get.return_value = collection_mock
        collection_mock.query.near_vector.return_value.objects = []

        ctx = manager.read_context("nonexistent-thread", _make_fake_embedding())

        assert isinstance(ctx, MemoryContext)
        assert ctx.history == []
        assert ctx.semantic_results == []
        assert ctx.patterns == []
        assert ctx.summary is None

    def test_read_context_with_history(self, manager, mock_weaviate, db_session):
        thread_id = str(uuid.uuid4())

        # Write some turns directly
        from src.memory.conversational import save_turn

        save_turn(db_session, thread_id, "user", "Hello")
        save_turn(db_session, thread_id, "assistant", "Hi there")

        # Mock Weaviate
        collection_mock = MagicMock()
        mock_weaviate.collections.get.return_value = collection_mock
        collection_mock.query.near_vector.return_value.objects = []

        ctx = manager.read_context(thread_id, _make_fake_embedding())

        assert len(ctx.history) == 2
        assert ctx.history[0].content == "Hello"
        assert ctx.history[1].content == "Hi there"

    def test_read_context_with_semantic_results(self, manager, mock_weaviate):
        # Mock Weaviate semantic search with results
        sem_collection = MagicMock()
        proc_collection = MagicMock()

        def get_collection(name):
            if name == "SemanticMemory":
                return sem_collection
            return proc_collection

        mock_weaviate.collections.get.side_effect = get_collection

        mock_obj = MagicMock()
        mock_obj.uuid = uuid.uuid4()
        mock_obj.properties = {"content": "PIX is free", "source": "agent", "thread_id": "t1"}
        mock_obj.metadata.distance = 0.2
        sem_collection.query.near_vector.return_value.objects = [mock_obj]
        proc_collection.query.near_vector.return_value.objects = []

        ctx = manager.read_context("thread-1", _make_fake_embedding())

        assert len(ctx.semantic_results) == 1
        assert ctx.semantic_results[0].content == "PIX is free"


class TestWriteTurn:
    def test_write_turn_persists(self, manager, db_session):
        thread_id = str(uuid.uuid4())
        turn = manager.write_turn(thread_id, "user", "Test message")

        assert isinstance(turn, ConversationTurn)
        assert turn.content == "Test message"

    def test_write_turn_with_metadata(self, manager):
        thread_id = str(uuid.uuid4())
        turn = manager.write_turn(thread_id, "user", "Hi", metadata={"key": "val"})

        assert turn.metadata_json == '{"key": "val"}'

    def test_auto_summarize_triggers_at_threshold(self, manager, db_session):
        thread_id = str(uuid.uuid4())
        # threshold=5, so write 5 turns to trigger
        for i in range(5):
            manager.write_turn(thread_id, "user", f"Turn {i}")

        from src.memory.summary import get_summary

        result = get_summary(db_session, thread_id)
        assert result is not None
        assert "Conversation summary" in result

    def test_no_summarize_below_threshold(self, manager, db_session):
        thread_id = str(uuid.uuid4())
        for i in range(3):
            manager.write_turn(thread_id, "user", f"Turn {i}")

        from src.memory.summary import get_summary

        assert get_summary(db_session, thread_id) is None


class TestWriteSemantic:
    def test_write_semantic_delegates(self, manager, mock_weaviate):
        collection_mock = MagicMock()
        mock_weaviate.collections.get.return_value = collection_mock

        object_id = manager.write_semantic(
            content="Important fact",
            embedding=_make_fake_embedding(),
            source="agent",
            thread_id="t1",
        )

        assert isinstance(object_id, str)
        collection_mock.data.insert.assert_called_once()


class TestWritePattern:
    def test_write_pattern_delegates(self, manager, mock_weaviate):
        collection_mock = MagicMock()
        mock_weaviate.collections.get.return_value = collection_mock

        object_id = manager.write_pattern(
            trigger="user asks about fees",
            action="check fee schedule",
            embedding=_make_fake_embedding(),
        )

        assert isinstance(object_id, str)
        collection_mock.data.insert.assert_called_once()
